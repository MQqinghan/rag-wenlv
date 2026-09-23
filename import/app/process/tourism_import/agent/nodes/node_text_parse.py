from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.tourism_import.agent.state import TourismImportGraphState
from app.rag.common.text_parse_service import parse_text_to_markdown


@node_log("node_text_parse")
def node_text_parse(state: TourismImportGraphState) -> TourismImportGraphState:
    """
    节点: 文本类解析 (node_text_parse)
    为什么叫这个名字: txt/json/html/docx → Markdown（复用标题切分管线），无图片处理。
    """
    add_running_task(state["task_id"], "node_text_parse")
    state = parse_text_to_markdown(state)
    add_done_task(state["task_id"], "node_text_parse")
    return state
