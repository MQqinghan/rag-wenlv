"""
文旅导入流程节点：表格（Excel/CSV）解析。
每行序列化为 "字段名: 值" 的文本，天然是一块完整语义，无需标题切分。
本节点产物 chunks 直接进入下游元数据抽取，跳过 node_document_split。
"""
from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.tourism_import.agent.state import TourismImportGraphState
from app.rag.common.tabular_parse_service import parse_table_to_chunks


@node_log("node_tabular_parse")
def node_tabular_parse(state: TourismImportGraphState) -> TourismImportGraphState:
    """
    节点: 表格解析 (node_tabular_parse)
    为什么叫这个名字: 把 Excel/CSV 表格解析成行级 chunk，跳过 PDF/MD 解析与标题切分。
    """
    add_running_task(state["task_id"], "node_tabular_parse")
    state = parse_table_to_chunks(state)
    add_done_task(state["task_id"], "node_tabular_parse")
    return state


if __name__ == "__main__":
    import os
    import tempfile
    from dotenv import load_dotenv
    from app.shared.utils.path_util import PROJECT_ROOT
    from app.process.tourism_import.agent.state import create_default_state
    import pandas as pd

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    logger.info("===== tourism node_tabular_parse 单元测试 =====")

    # 构造临时 xlsx 测试
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        xlsx_path = f.name
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame({"景点名称": ["兵马俑", "华清池"], "等级": ["5A", "5A"], "门票": ["120元", "120元"]}).to_excel(writer, sheet_name="西安", index=False)
        state = create_default_state(task_id="t_tab_001", local_file_path=xlsx_path)
        result = node_tabular_parse(state)
        chunks = result.get("chunks", [])
        logger.info(f"解析出 {len(chunks)} 条 chunk（应=2）")
        assert len(chunks) == 2
        assert "兵马俑" in chunks[0]["content"]
        logger.info(f"首条 chunk: {chunks[0]['content']}")
    finally:
        os.unlink(xlsx_path)
    logger.info("===== tourism node_tabular_parse 测试通过 =====")
