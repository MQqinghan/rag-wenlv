"""
文旅元数据抽取服务模块
负责从文档切片中抽取文旅元数据（主体名/内容类型/地区/文化主题/类别/专属字段），
回填到切片，并同步生成主体向量写入 Milvus tourism_entity 主体索引。
一次 LLM 调用同时完成分类 + 结构化抽取；失败降级为"运营资料"，保证导入不中断。
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from app.rag.tourism_import.content_schema import (
    TourismMetadata,
    TourismContentType,
    normalize_extra,
)
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider, invoke_llm_with_retry
from app.rag.tourism_import.index_service import upsert_entity
from app.shared.utils.city_utils import extract_cities
from app.rag.common.chunk_config import (
    ITEM_NAME_CONTEXT_CHUNK_K,
    ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS,
)


@step_log("validate_chunks_and_title")
def validate_chunks_and_title(state: dict) -> tuple[list[dict], str]:
    """校验并提取 state 中的 chunks 和 file_title。"""
    chunks = state.get("chunks", [])
    file_title = state.get("file_title")

    if not chunks:
        logger.error("chunks没有内容,无法继续业务!")
        raise ValueError("chunks没有内容,无法继续业务!")

    if not file_title:
        logger.warning("file_title为空给与默认值处理!")
        file_title = "default_title"
        state["file_title"] = file_title

    return chunks, file_title


@step_log("build_document_context")
def build_document_context(chunks: list[dict]) -> str:
    """从前 K 个切片构建用于 LLM 识别的上下文字符串。"""
    current_chunks = chunks[:ITEM_NAME_CONTEXT_CHUNK_K]
    chunk_str_list: list[str] = []
    for index, item in enumerate(current_chunks, start=1):
        chunk_str_list.append(f"切片:{index},标题:{item.get('title','')},内容:{item.get('content','')}")
    chunk_str = "\n".join(chunk_str_list)
    return chunk_str[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]


@step_log("parse_metadata")
def parse_metadata(raw: str, file_title: str) -> TourismMetadata:
    """
    解析 LLM 输出，失败时降级兜底，保证流程不中断。
    降级策略：content_type=运营资料，item_name=file_title。
    解析成功后对 extra 做键名白名单清洗（剔除 LLM 误输出的说明文字/类型名等脏键）。
    """
    try:
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        meta = TourismMetadata.model_validate_json(raw)
        if not meta.item_name:
            meta.item_name = file_title
        # 清洗 extra：按 content_type 专属字段模型过滤键名（无专属模型的类型清空）
        meta.extra = normalize_extra(meta.content_type, meta.extra)
        return meta
    except Exception as e:
        # 异常文本用位置参数传递，避免其中的花括号被 loguru 当占位符解析（KeyError）
        logger.warning("元数据 JSON 解析失败，降级处理：{}", repr(e))
        return TourismMetadata(
            item_name=file_title,
            content_type=TourismContentType.OPERATION_MATERIAL,
        )


@step_log("extract_metadata")
def extract_metadata(context: str, file_title: str) -> TourismMetadata:
    """调用 LLM 抽取文旅元数据（JSON 输出，限流时指数退避重试）。"""
    llm = llm_provider.chat()
    user_prompt = load_prompt("tourism/tourism_meta_extract", file_title=file_title, context=context)
    raw = invoke_llm_with_retry(llm | StrOutputParser(), [HumanMessage(content=user_prompt)])
    return parse_metadata(raw, file_title)


@step_log("build_meta_from_forced_type")
def build_meta_from_forced_type(forced_type: str, file_title: str) -> TourismMetadata:
    """
    前端手动指定内容类型时，跳过 LLM 分类，直接构造元数据（降本）。
    item_name 用 file_title 兜底，其余字段留空，由后续检索按需补全。
    """
    try:
        ct = TourismContentType(forced_type)
    except ValueError:
        logger.warning(f"前端指定类型[{forced_type}]不在枚举中，降级为运营资料")
        ct = TourismContentType.OPERATION_MATERIAL
    return TourismMetadata(item_name=file_title, content_type=ct)


@step_log("apply_metadata")
def apply_metadata(chunks: list[dict], meta: TourismMetadata, source_type: str, source_path: str = "") -> list[dict]:
    """元数据回填到每个 chunk，字段与 tourism_chunks 集合显式字段一一对应。

    source_path 为上传文件本地路径（对齐需求"来源路径或资源链接"）。
    """
    for chunk in chunks:
        chunk["item_name"] = meta.item_name
        chunk["content_type"] = meta.content_type.value
        chunk["region"] = meta.region or ""
        chunk["cultural_theme"] = meta.cultural_theme or ""
        chunk["category"] = meta.category or ""
        chunk["source_type"] = source_type
        chunk["extra_meta"] = meta.extra or {}
        chunk["source_path"] = source_path
    return chunks


@step_log("extract_and_index_tourism_metadata")
def extract_and_index_tourism_metadata(state: dict) -> dict:
    """
    文旅元数据抽取服务总入口：
    抽取 → 回填 state 和 chunks → 主体向量化 → 写入 Milvus tourism_entity。
    支持前端 forced_content_type 跳过 LLM 分类降本。
    """
    # 1. 校验输入
    chunks, file_title = validate_chunks_and_title(state)

    # 2. 抽取元数据（有 forced_type 走降本路径，否则 LLM 抽取）
    forced_type = state.get("forced_content_type") or ""
    if forced_type:
        logger.info(f"前端指定内容类型[{forced_type}]，跳过 LLM 分类（降本）")
        meta = build_meta_from_forced_type(forced_type, file_title)
    else:
        context = build_document_context(chunks)
        meta = extract_metadata(context, file_title)

    # B2 城市标签兜底（2026-09-11 主人拍板方案二）：LLM 抽取 region 为空时，
    # 用共享层城市名单从文件名/主体名/正文做确定性回填（存量实测 55.6% region 为空，
    # 检索侧按 region 过滤依赖该字段的覆盖率）。多城市以逗号连接（双城攻略场景）。
    if not meta.region:
        sample_text = " ".join([
            file_title or "",
            meta.item_name or "",
            (chunks[0].get("title") or "") if chunks else "",
            (chunks[0].get("content") or "")[:300] if chunks else "",
        ])
        cities = extract_cities(sample_text)
        if cities:
            meta.region = ",".join(cities)
            logger.info(f"B2 region 兜底回填：{meta.region}（城市名单确定性匹配）")

    logger.info(
        f"文旅元数据抽取完成：file_title={file_title}，item_name={meta.item_name}，"
        f"type={meta.content_type.value}，region={meta.region or '(空)'}"
    )

    # 3. 写回 state 和 chunks（source_path=上传文件本地路径，对齐需求"来源路径或资源链接"）
    state["item_name"] = meta.item_name
    state["content_type"] = meta.content_type.value
    state["tourism_meta"] = meta.model_dump()
    source_type = state.get("source_type") or _infer_source_type(state)
    state["chunks"] = apply_metadata(chunks, meta, source_type, state.get("local_file_path") or "")

    # 4. 主体向量化 + 入库
    # 主体（景点/城市）是检索入口（意图路由、item_name 过滤依赖 entity 集合），
    # 入库失败必须终止任务，否则会产生"chunks 有数据、entity 没数据"的半成品导入。
    result = llm_provider.embed_documents([meta.item_name])
    dense_vector, sparse_vector = result["dense"][0], result["sparse"][0]
    upsert_entity(meta, file_title, dense_vector, sparse_vector, state.get("local_file_path") or "")

    return state


def _infer_source_type(state: dict) -> str:
    """从文件名推断来源类型（pdf/md/xlsx/csv）。"""
    file_path = state.get("local_file_path") or ""
    if "." in file_path:
        return file_path.rsplit(".", 1)[-1].lower()
    return "pdf"


if __name__ == "__main__":
    # 单元测试：mock 文旅 chunks 验证抽取（需 LLM + BGE-M3）
    import os
    from dotenv import load_dotenv

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    load_dotenv(os.path.join(project_root, ".env"))

    logger.info("===== tourism_meta_extract_service 单元测试 =====")
    mock_state = {
        "task_id": "test_tourism_meta_001",
        "file_title": "西安景点资料",
        "local_file_path": "西安景点资料.pdf",
        "chunks": [
            {
                "title": "景点简介",
                "content": "兵马俑位于陕西西安，是秦始皇陵陪葬坑，国家5A级景区，门票120元，开放时间8:30-17:30。",
            },
            {
                "title": "游览建议",
                "content": "建议游览时长3小时，最佳季节春秋两季，可乘坐地铁9号线到达。",
            },
        ],
    }

    try:
        result_state = extract_and_index_tourism_metadata(mock_state)
        logger.info(f"主体名：{result_state.get('item_name')}")
        logger.info(f"元数据：{result_state.get('tourism_meta')}")
        first_chunk = result_state.get("chunks", [{}])[0]
        logger.info(f"首切片 item_name：{first_chunk.get('item_name')}")
        logger.info(f"首切片 content_type：{first_chunk.get('content_type')}")
        logger.info(f"首切片 source_type：{first_chunk.get('source_type')}")
    except Exception as e:
        logger.error(f"抽取测试失败：{e}", exc_info=True)

    # 测试2：forced_content_type 降本路径（不调 LLM 分类）
    logger.info("--- 测试 forced_content_type 降本路径 ---")
    mock_state2 = {
        "task_id": "test_tourism_meta_002",
        "file_title": "景点清单",
        "local_file_path": "景点清单.xlsx",
        "forced_content_type": "景点信息",
        "chunks": [{"title": "行1", "content": "景点名称: 兵马俑, 等级: 5A"}],
    }
    try:
        result2 = extract_and_index_tourism_metadata(mock_state2)
        logger.info(f"降本路径 item_name：{result2.get('item_name')}")
        logger.info(f"降本路径 content_type：{result2.get('content_type')}")
        assert result2.get("content_type") == "景点信息"
    except Exception as e:
        logger.error(f"降本路径测试失败：{e}", exc_info=True)

    logger.info("===== tourism_meta_extract_service 测试结束 =====")
