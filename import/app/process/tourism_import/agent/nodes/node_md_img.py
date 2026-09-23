"""文旅导入流程节点：Markdown 图片多模态增强（复用 common 服务）。"""
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.tourism_import.agent.state import TourismImportGraphState
from app.rag.common.markdown_image_service import enrich_markdown_images


@node_log("node_md_img")
def node_md_img(state: TourismImportGraphState) -> TourismImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    """
    add_running_task(state["task_id"], "node_md_img")
    state = enrich_markdown_images(state)
    add_done_task(state["task_id"], "node_md_img")
    return state


if __name__ == "__main__":
    import os
    from loguru import logger
    from app.shared.utils.path_util import PROJECT_ROOT

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")
    test_md_path = os.path.join(PROJECT_ROOT, "output", "兵马俑", "兵马俑.md")
    if not os.path.exists(test_md_path):
        logger.error(f"测试文件不存在：{test_md_path}，请将测试 MD 放入 output 目录")
    else:
        test_state = {"md_path": test_md_path, "task_id": "t_md_img_001", "md_content": ""}
        result_state = node_md_img(test_state)
        logger.info(f"MD 图片处理完成，结果状态：{result_state}")
