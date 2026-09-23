"""文旅导入流程节点：文档切分（复用 common 切分服务）。"""
import os
from loguru import logger
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.tourism_import.agent.state import TourismImportGraphState
from app.rag.common.split_service import split_document


@node_log("node_document_split")
def node_document_split(state: TourismImportGraphState) -> TourismImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    """
    add_running_task(state["task_id"], "node_document_split")
    state = split_document(state)
    add_done_task(state["task_id"], "node_document_split")
    return state


if __name__ == "__main__":
    from app.shared.utils.path_util import PROJECT_ROOT
    from app.process.tourism_import.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")
    test_md_path = os.path.join(PROJECT_ROOT, "output", "兵马俑", "兵马俑.md")
    if not os.path.exists(test_md_path):
        logger.error(f"测试文件不存在：{test_md_path}，请将测试 MD 放入 output 目录")
    else:
        test_state = {
            "md_path": test_md_path,
            "task_id": "t_split_001",
            "md_content": "",
            "file_title": "兵马俑",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        result_state = node_md_img(test_state)
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"测试成功：最终生成 {len(final_chunks)} 个有效 Chunk")
