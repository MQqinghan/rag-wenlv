import json
from pathlib import Path
from app.shared.runtime.logger import logger, node_log, step_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.rag.common.chunk_config import (
    SUPPORTED_TABLE_EXTENSIONS,
    SUPPORTED_TEXT_EXTENSIONS,
)


@step_log("resolve_input_file")
def resolve_input_file(state: dict) -> dict:
    """
    文件类型识别与状态初始化节点（导入流程入口）。
    通用：按后缀决定走 md / pdf / 表格 / 纯文本 四条路径，非法类型终止流程。
    不绑定具体业务 state（文旅状态为 dict 超集），保证复用。
    """
    # 1. 获取文件本地路径
    local_file_path = state.get("local_file_path")

    # 2. 校验文件路径是否为空，为空则直接结束流程
    if not local_file_path:
        logger.warning("节点:resolve_input_file, 文件路径为空，直接终止当前导入流程")
        return state

    # 3. 统一初始化四个路由开关，再按后缀打开其中一个
    state["is_md_read_enabled"] = False
    state["is_pdf_read_enabled"] = False
    state["is_table_read_enabled"] = False
    state["is_text_read_enabled"] = False

    # 4. 识别文件类型并设置对应状态与路由开关
    if local_file_path.endswith(".md"):
        state["md_path"] = local_file_path
        state["is_md_read_enabled"] = True

    elif local_file_path.endswith(".pdf"):
        state["pdf_path"] = local_file_path
        state["is_pdf_read_enabled"] = True

    elif Path(local_file_path).suffix.lower() in SUPPORTED_TABLE_EXTENSIONS:
        # 表格类（xlsx/xls/csv）：直接走表格解析分支，跳过 PDF/MD 解析与切分
        state["is_table_read_enabled"] = True

    elif Path(local_file_path).suffix.lower() in SUPPORTED_TEXT_EXTENSIONS:
        # 纯文本类（txt/json）：转 Markdown 后复用标题切分管线，跳过图片处理
        state["is_text_read_enabled"] = True

    else:
        # 不支持的文件类型：显式抛错，让任务标记失败（避免静默"completed"误导用户）
        raise ValueError(
            f"不支持的文件类型: {Path(local_file_path).suffix or '无后缀'}，"
            f"支持 PDF/MD/DOCX/TXT/JSON/HTML/XLSX/CSV 等文本/表格类文件"
        )

    # 5. 自动提取文件标题（不带后缀）
    state["file_title"] = Path(local_file_path).stem

    # 6. 返回补全后的状态
    return state