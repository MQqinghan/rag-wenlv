"""
文旅导入流程节点：PDF 转 Markdown（复用 MinerU 解析服务）。
"""
import os

from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.tourism_import.agent.state import TourismImportGraphState, create_default_state
from app.rag.common.pdf_parse_service import parse_pdf_to_markdown
from app.shared.utils.path_util import PROJECT_ROOT


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: TourismImportGraphState) -> TourismImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    """
    add_running_task(state["task_id"], "node_pdf_to_md")
    state = parse_pdf_to_markdown(state)
    add_done_task(state["task_id"], "node_pdf_to_md")
    return state


if __name__ == "__main__":
    logger.info("===== tourism node_pdf_to_md 单元测试 =====")
    test_pdf_name = os.path.join(r"doc", "兵马俑.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)
    test_state = create_default_state(
        task_id="t_pdf2md_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output"),
    )
    if not os.path.exists(test_pdf_path):
        logger.error(f"测试文件不存在：{test_pdf_path}，请将测试 PDF 放入 doc 目录")
    else:
        node_pdf_to_md(test_state)
    logger.info("===== tourism node_pdf_to_md 测试结束 =====")
