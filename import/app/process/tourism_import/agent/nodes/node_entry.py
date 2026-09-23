from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.tourism_import.agent.state import TourismImportGraphState
from app.rag.common.entry_service import resolve_input_file


@node_log("node_entry")
def node_entry(state: TourismImportGraphState) -> TourismImportGraphState:
    """
    节点: 入口识别 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，按文件后缀决定走 md / pdf / 表格 三条路径。
    """
    add_running_task(state["task_id"], "node_entry")
    state = resolve_input_file(state)
    add_done_task(state["task_id"], "node_entry")
    return state


if __name__ == "__main__":
    import json
    from app.shared.runtime.logger import logger
    from app.process.tourism_import.agent.state import create_default_state

    logger.info("===== tourism node_entry 单元测试 =====")
    # 测试1: PDF
    s1 = create_default_state(task_id="t_entry_1", local_file_path="兵马俑.pdf")
    r1 = node_entry(s1)
    logger.info(f"PDF 路由: pdf={r1['is_pdf_read_enabled']}, table={r1['is_table_read_enabled']}")
    assert r1["is_pdf_read_enabled"] and not r1["is_table_read_enabled"]

    # 测试2: MD
    s2 = create_default_state(task_id="t_entry_2", local_file_path="北欧文化.md")
    r2 = node_entry(s2)
    assert r2["is_md_read_enabled"]

    # 测试3: 表格 xlsx
    s3 = create_default_state(task_id="t_entry_3", local_file_path="景点清单.xlsx")
    r3 = node_entry(s3)
    logger.info(f"XLSX 路由: table={r3['is_table_read_enabled']}")
    assert r3["is_table_read_enabled"]

    # 测试4: CSV
    s4 = create_default_state(task_id="t_entry_4", local_file_path="data.csv")
    r4 = node_entry(s4)
    assert r4["is_table_read_enabled"]

    # 测试5: 不支持类型
    s5 = create_default_state(task_id="t_entry_5", local_file_path="xxx.txt")
    r5 = node_entry(s5)
    assert not (r5["is_md_read_enabled"] or r5["is_pdf_read_enabled"] or r5["is_table_read_enabled"])
    logger.info("===== tourism node_entry 测试通过 =====")
