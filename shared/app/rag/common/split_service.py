import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.shared.runtime.logger import logger, step_log
from app.rag.common.chunk_config import CHUNK_MAX_SIZE, CHUNK_SIZE


@step_log("split_document")
def split_document(state: dict) -> dict:
    """
    文档切块核心节点（RAG 最关键步骤）
    功能：加载增强后的 Markdown 内容 → 按标题智能切块 → 优化块大小 → 备份切块结果 → 写入状态
    输出：将分块后的文本列表存入 state，供后续向量化、入库使用
    """
    # 1. 从状态中加载【增强后的Markdown内容】和【文档标题】
    md_content, file_title = load_markdown_content(state)

    # 2. 按 Markdown 标题（#、##、###）进行【智能语义切块】（保持段落完整性）
    chunks = split_by_titles(md_content, file_title)

    # 3. 优化切块：对过长的块再次拆分、过短的块过滤，保证符合向量库入库标准
    # CHUNK_MAX_SIZE：块最大长度；CHUNK_SIZE：块最小有效长度
    chunks = refine_chunks(chunks, max_len=CHUNK_MAX_SIZE, min_len=CHUNK_SIZE)

    # 4. 备份切块结果到本地文件（方便调试、审计、回溯）
    backup_chunks(chunks, state["md_path"])

    # 5. 将最终合格的切块列表写入流程状态，传递给下游节点（向量化/入库）
    state["chunks"] = chunks

    # 6. 返回更新后的状态
    return state

@step_log("load_markdown_content")
def load_markdown_content(state: dict) -> tuple[str, str]:
    """
    从状态字典中安全加载 Markdown 内容和文档标题
    1. 优先从 state 中直接读取
    2. 缺失时自动从文件读取兜底
    3. 统一换行符格式，保证文本干净
    :return: (处理后的md内容, 文件标题)
    """
    # 从状态中获取核心数据：md内容、文件标题、md文件路径
    md_content = state.get("md_content", "")
    file_title = state.get("file_title", "")
    md_path = state.get("md_path", "")

    # ===================== 处理 md_content 缺失场景 =====================
    # 如果状态中没有md内容，尝试从本地md文件读取（兜底逻辑）
    if not md_content:
        logger.warning("没有从state读取到md_content内容,我们使用md_path尝试再次读取!")

        # 如果文件路径存在，则读取文件内容
        if md_path:
            md_content = Path(md_path).read_text(encoding="utf-8")
            state["md_content"] = md_content  # 读取后回填到状态，避免重复读取

        # 双重校验：仍然无内容，直接抛出异常终止流程
        if not md_content:
            raise ValueError("md_content没数据,并且尝试读取md_path依然没有数据,终止执行!!")

    # ===================== 处理 file_title 缺失场景 =====================
    # 如果标题为空，使用文件名（无后缀）作为标题；无路径则使用默认值
    if not file_title:
        file_title = Path(md_path).stem if md_path else "default"
        state["file_title"] = file_title  # 回填到状态

    # ===================== 统一文本格式 =====================
    # 替换所有换行符为 \n，解决 Windows/Linux 换行符不一致问题
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    # 返回处理好的文本内容 + 标题，给后续切块使用
    return md_content, file_title


@step_log("split_by_titles")
def split_by_titles(md_content: str, file_title: str) -> list[dict]:
    """
    按 Markdown 标题（#、##、###...）进行【语义化文档切块】
    特点：
        1. 自动识别标题，保证段落语义完整
        2. 跳过代码块内部的内容，不把 ``` 内的内容误判为标题
        3. 每个块包含：内容、当前标题、文档标题，方便后续检索
    :param md_content: Markdown 文本内容
    :param file_title: 文档名称（用于溯源）
    :return: 切块列表，每个元素是 {content, title, file_title}
    """
    # 正则：匹配 Markdown 标题（# ~ ###### 开头的行）
    reg = re.compile(r"^\s*#{1,6}\s.+")

    # 将全文按换行符切割成逐行处理
    lines = md_content.split("\n")

    # 存储最终切块结果
    chunks: list[dict] = []

    # 当前正在拼接的块标题
    current_title = None

    # 当前块的所有行内容
    current_title_lines: list[str] = []

    # 标记：是否处于代码块（```...```）内部
    is_code_block = False

    # 记录切块数量
    chunk_size = 0

    # 逐行遍历 MD 内容
    for raw_line in lines:
        line = raw_line.strip()

        # ===================== 代码块判断 =====================
        # 遇到 ``` 或 ~~~ 标记，切换代码块状态
        if line.startswith("```") or line.startswith("~~~"):
            is_code_block = not is_code_block
            current_title_lines.append(line)  # 把代码行加入当前块
            continue

        # ===================== 识别标题并切分 =====================
        # 如果当前行是标题，并且**不在代码块内**，才进行切分
        if reg.match(line) and not is_code_block:
            # 如果已有上一个块内容，就把上一个块保存
            if current_title and len(current_title_lines) > 1:
                chunks.append({
                    "content": "\n".join(current_title_lines),  # 块内容
                    "title": current_title,  # 块标题
                    "file_title": file_title  # 文档名（溯源用）
                })

            # 以当前行作为新块的标题
            current_title = line
            current_title_lines = [current_title]
            chunk_size += 1

        # 普通行 → 直接追加到当前块
        else:
            current_title_lines.append(line)

    # ===================== 保存最后一个块 =====================
    if current_title:
        chunks.append({
            "content": "\n".join(current_title_lines),
            "title": current_title,
            "file_title": file_title
        })

    # ===================== 兜底：全文无标题时 =====================
    if chunk_size == 0:
        chunks.append({
            "content": md_content,
            "title": "default",
            "file_title": file_title
        })

    return chunks

@step_log("refine_chunks")
def refine_chunks(
    sections: list[dict],
    max_len: int = CHUNK_MAX_SIZE,
    min_len: int = CHUNK_SIZE,
) -> list[dict]:
    """
    【切块精细化处理】RAG 核心优化步骤
    作用：把按标题切好的块，进一步调整成【长度标准、适合入库】的块
    流程：
        1. 太长的块 → 拆分（不超过 max_len）
        2. 太短的块 → 合并（不低于 min_len）
        3. 统一补全字段（part、parent_title）
    返回：最终标准、可用的切块列表
    """
    # 如果最大长度配置无效，直接返回原始切块，不做处理
    if not max_len or max_len <= 0:
        logger.warning(f"步骤4：Chunk最大长度配置无效（{max_len}），跳过精细化处理")
        return sections

    # 存储处理后的中间切块
    refined_split = []

    # 遍历所有按标题切好的块
    for sec in sections:
        # 拆分过长的块，并加入结果列表
        refined_split.extend(_split_long_section(sec, max_len))

    # 合并过短的块，得到最终符合长度要求的切块
    final_sections = _merge_short_sections(refined_split, min_length=min_len, max_length=max_len)

    # 统一给所有块补全字段（方便后续检索、溯源、展示）
    for sec in final_sections:
        # 给切块编号：同一标题下的第 N 部分，默认 0
        if "part" not in sec:
            sec["part"] = 0
        # 记录父标题：用于溯源属于哪个大章节
        if not sec.get("parent_title"):
            sec["parent_title"] = sec.get("title") or ""

    # 返回最终标准化的切块
    return final_sections

def _split_long_section(section: dict[str, Any], max_length: int = CHUNK_MAX_SIZE) -> list[dict[str, Any]]:
    """
    内部工具函数：拆分【过长的文本块】，保证单个chunk不超过最大长度限制
    核心逻辑：
        1. 检查内容长度，不长则直接返回
        2. 标题单独保留，只拆分正文内容
        3. 使用语义化拆分器，按段落、句子拆分，保证语义完整
    :param section: 待拆分的切块（包含title、content等）
    :param max_length: 单个块最大字符长度
    :return: 拆分后的子块列表
    """
    # 获取块的正文内容
    content = section.get("content", "") or ""

    # 如果内容长度未超限，无需拆分，直接返回原块
    if len(content) <= max_length:
        return [section]

    # 统一换行符格式，避免不同系统换行符导致拆分异常
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    # 获取当前块的标题
    title = section.get("title", "") or ""

    # 拼接标题前缀（标题+换行），这部分会占用长度，需要预留空间
    prefix = f"{title}\n\n" if title else ""
    # 计算正文可使用的最大长度（总长度 - 标题占用长度）
    available_len = max_length - len(prefix)

    # 如果预留后正文无可用长度，直接返回原块，不拆分
    if available_len <= 0:
        return [section]

    # 提取纯正文内容：如果正文以标题开头，剔除标题部分，只保留内容
    body = content
    if title and body.lstrip().startswith(title):
    	body = body[len(title):].lstrip()  # 直接切掉标题长度

    # 初始化递归字符拆分器（LangChain官方工具）
    # 按 段落→换行→句子→空格 优先级拆分，保证语义完整性
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len,       # 拆分后的正文最大长度
        chunk_overlap=0,                # 块之间无重叠
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
    )

    sub_sections = []
    # 遍历拆分后的正文片段，生成子块
    for idx, chunk in enumerate(splitter.split_text(body), start=1):
        text = chunk.strip()
        # 跳过空内容
        if not text:
            continue

        # 拼接完整内容：标题 + 拆分后的正文
        full_text = (prefix + text).strip()

        # 构造子块，保留溯源信息，添加分区编号
        sub_sections.append({
            "title": f"{title}-{idx}" if title else f"chunk-{idx}",  # 子标题：原标题-序号
            "content": full_text,                                    # 完整内容
            "parent_title": title,                                   # 父标题（用于溯源）
            "part": idx,                                             # 序号（同一章节下的第N部分）
            "file_title": section.get("file_title"),                 # 文档原始标题
        })

    # 返回拆分完成的所有子块
    return sub_sections

def _merge_short_sections(
    sections: list[dict[str, Any]],
    min_length: int = CHUNK_SIZE,
    max_length: int = CHUNK_MAX_SIZE,
) -> list[dict[str, Any]]:
    """
    内部工具函数：合并【过短的文本块】，避免碎片内容
    核心规则：
        1. 只有长度 < 最小长度 的短块才会被合并
        2. 必须是**同一个父标题/章节**下的内容才会合并（保证语义相关）
        3. 合并后不能超过最大长度，避免再次超长
        4. 自动去重重复标题，保证内容干净
    :param sections: 待合并的文本块列表
    :param min_length: 最小长度阈值，低于此值视为短块
    :param max_length: 最大长度阈值，合并后不能超限
    :return: 合并完成后的规整文本块列表
    """
    # 空列表直接返回
    if not sections:
        return []

    # 存储最终合并完成的块
    merged_sections = []
    # 当前正在累积合并的块
    current_chunk = None

    # 遍历所有待处理的块
    for sec in sections:
        # 初始化：第一个块直接作为当前块
        if current_chunk is None:
            current_chunk = sec
            continue

        # 获取当前块的内容
        current_content = current_chunk.get("content", "")
        # 判断：当前块是否过短
        is_current_short = len(current_content) < min_length
        # 判断：当前块 和 下一个块 是否属于同一个父章节（保证语义相关）
        is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

        # ===================== 满足条件：执行合并 =====================
        if is_current_short and is_same_parent:
            # 获取父标题，用于去重
            parent_title = sec.get("parent_title", "")
            next_content = sec["content"]

            # 如果下一个块内容以父标题开头，剔除重复标题，避免冗余
            if parent_title and next_content.startswith(parent_title):
                next_content = next_content[len(parent_title):].lstrip()

            # 拼接合并后的内容
            merged_content = current_content + "\n\n" + next_content
            # 判断：合并后是否会超过最大长度限制
            will_exceed_max = max_length > 0 and len(merged_content) > max_length

            # 如果合并会超长 → 不合并，保存当前块，将下一个块作为新的当前块
            if will_exceed_max:
                merged_sections.append(current_chunk)
                current_chunk = sec
                continue

            # 执行合并：更新当前块内容
            current_chunk["content"] = merged_content
            # 同步序号信息（可选，用于溯源）
            if "part" in sec:
                current_chunk["part"] = sec["part"]

        # ===================== 不满足合并条件：直接保存当前块 =====================
        else:
            merged_sections.append(current_chunk)
            current_chunk = sec

    # 循环结束，把最后一个当前块加入结果
    if current_chunk is not None:
        merged_sections.append(current_chunk)

    # 返回合并完成的规整块列表
    return merged_sections


@step_log("backup_chunks")
def backup_chunks(chunks: list[dict], md_path: str) -> None:
    """
    备份文档切块结果到本地 JSON 文件
    作用：持久化保存切块数据，方便调试、校验、回溯，避免重复处理
    :param chunks: 文档切块后的列表数据
    :param md_path: 原始 Markdown 文件路径（用于确定保存目录）
    """
    # 在 MD 文件所在目录下，生成 chunks.json 保存路径
    chunks_json_path = Path(md_path).parent / "chunks.json"

    # 将切块列表转为格式化 JSON 并写入文件
    # ensure_ascii=False：保证中文正常显示不转义
    # indent=4：格式化缩进，方便人工查看
    chunks_json_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=4), encoding="utf-8")

    # with open(chunks_json_path, "w", encoding="utf-8") as f:
    # json.dump(chunks, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # 测试 refine_chunks 的拆分与合并逻辑
    test_sections = [
        {
            "title": "## 功能说明",
            "content": "## 功能说明\n\n" + "A" * 1200,
            "file_title": "demo",
        },
        {
            "title": "## 参数说明",
            "content": "## 参数说明\n\n参数1：xxx",
            "file_title": "demo",
        },
    ]

    result = refine_chunks(test_sections, max_len=1000, min_len=600)

    print(f"\n===== refine_chunks 测试结果 =====")
    print(f"输入块数: {len(test_sections)}")
    print(f"输出块数: {len(result)}")
    for i, chunk in enumerate(result):
        print(f"\n--- Chunk {i} ---")
        print(f"  title:        {chunk.get('title')}")
        print(f"  parent_title: {chunk.get('parent_title')}")
        print(f"  part:         {chunk.get('part')}")
        print(f"  content_len:  {len(chunk.get('content', ''))}")
        print(f"  content_preview: {chunk.get('content', '')[:80]}...")