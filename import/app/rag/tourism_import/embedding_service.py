"""
文旅向量化服务模块，负责为文档切块批量生成稠密与稀疏向量。
前缀拼接："主体:{item_name},类型:{content_type},内容:{content}"
——把文旅主体（景点/文化主题/游记标题）与内容类型压进向量表示，
提升"按景点名/文化主题"检索的命中率。
"""
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.rag.common.chunk_config import EMBEDDING_BATCH_SIZE


@step_log("require_chunks")
def require_chunks(state: dict) -> list[dict]:
    """校验导入状态中是否已经生成切块结果。"""
    chunks = state.get("chunks", [])
    if not chunks:
        logger.error("chunks为空,无法继续业务处理!")
        raise ValueError("chunks为空,无法继续业务处理!")
    return chunks


@step_log("embed_chunks")
def embed_chunks(chunks: list[dict], *, step: int = EMBEDDING_BATCH_SIZE) -> list[dict]:
    """
    批量为 chunks 生成 BGE-M3 稠密+稀疏向量。
    语义增强前缀：主体:{item_name},类型:{content_type},内容:{content}
    单批失败跳过继续。
    """
    chunks_vector: list[dict] = []
    total = len(chunks)
    for index in range(0, total, step):
        try:
            step_chunks = chunks[index:index + step]
            vector_str_list = []
            for item in step_chunks:
                item_name = item.get("item_name") or ""
                content_type = item.get("content_type") or ""
                content = item.get("content", "")
                # 语义增强前缀：把文旅主体信息与内容类型压进向量表示
                if item_name and content_type:
                    vector_str_list.append(f"主体:{item_name},类型:{content_type},内容:{content}")
                elif item_name:
                    vector_str_list.append(f"主体:{item_name},内容:{content}")
                else:
                    vector_str_list.append(content)
            result = llm_provider.embed_documents(vector_str_list)
            for i, chunk in enumerate(step_chunks):
                chunk_new = chunk.copy()
                chunk_new["dense_vector"] = result["dense"][i]
                chunk_new["sparse_vector"] = result["sparse"][i]
                chunks_vector.append(chunk_new)
        except Exception as exc:
            logger.warning(f"index={index}步骤,发生错误,跳过,继续生成向量!!,错误信息:{str(exc)}")
            continue
    return chunks_vector


@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: dict) -> dict:
    """向量化服务总入口：校验 chunks → 批量编码 → 写回 state。"""
    state["chunks"] = embed_chunks(require_chunks(state))
    return state


if __name__ == "__main__":
    # 单元测试：mock 文旅 chunks 验证向量化（需 BGE-M3 模型）
    import os
    from dotenv import load_dotenv

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    load_dotenv(os.path.join(project_root, ".env"))

    logger.info("===== tourism embedding_service 单元测试 =====")
    test_state = {
        "task_id": "test_tourism_embed_001",
        "chunks": [
            {
                "content": "兵马俑位于陕西西安，是秦始皇陵陪葬坑，5A景区，门票120元。",
                "title": "景点简介",
                "item_name": "兵马俑",
                "content_type": "景点信息",
                "file_title": "西安景点资料",
            },
            {
                "content": "北欧文化以极简、自然、平等为核心特点，受维京传统影响深远。",
                "title": "文化特点",
                "item_name": "北欧文化",
                "content_type": "文化知识介绍",
                "file_title": "北欧文化资料",
            },
        ],
    }

    try:
        result_state = generate_chunk_embeddings(test_state)
        result_chunks = result_state.get("chunks", [])
        logger.info(f"向量化完成，处理切片数：{len(result_chunks)}")
        if result_chunks:
            logger.info(f"首切片 dense_vector 维度：{len(result_chunks[0].get('dense_vector', []))}")
            logger.info(f"首切片 sparse_vector 键数：{len(result_chunks[0].get('sparse_vector', {}))}")
    except Exception as e:
        logger.error(f"向量化测试失败（检查 BGE-M3 模型路径/显存）: {e}", exc_info=True)
    logger.info("===== tourism embedding_service 测试结束 =====")
