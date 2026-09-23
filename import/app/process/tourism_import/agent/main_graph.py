"""
文旅导入流程图编排入口，负责组织节点执行顺序与分支流转。
图结构：
  node_entry ──(条件路由: md? pdf? 表格? 文本类? 其他→END)
     ├─ is_md_read_enabled   → node_md_img
     ├─ is_pdf_read_enabled  → node_pdf_to_md → node_md_img
     ├─ is_table_read_enabled → node_tabular_parse ──┐(跳过切分)
     └─ is_text_read_enabled → node_text_parse（txt/json/html/docx → md，docx/html 顺带抽内嵌图片）→ node_md_img
                                                       │
  node_md_img → node_document_split ───────────────────┤
                                                        ↓
                              node_tourism_meta_extract → node_bge_embedding → node_import_milvus → END
"""
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START

from app.shared.runtime.logger import logger
from app.process.tourism_import.agent.state import TourismImportGraphState, create_default_state
from app.process.tourism_import.agent.nodes.node_entry import node_entry
from app.process.tourism_import.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.tourism_import.agent.nodes.node_md_img import node_md_img
from app.process.tourism_import.agent.nodes.node_text_parse import node_text_parse
from app.process.tourism_import.agent.nodes.node_document_split import node_document_split
from app.process.tourism_import.agent.nodes.node_tabular_parse import node_tabular_parse
from app.process.tourism_import.agent.nodes.node_tourism_meta_extract import node_tourism_meta_extract
from app.process.tourism_import.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.tourism_import.agent.nodes.node_import_milvus import node_import_milvus

load_dotenv()

# 1. 定义状态图对象
workflow = StateGraph(TourismImportGraphState)

# 2. 添加节点
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_text_parse", node_text_parse)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_tabular_parse", node_tabular_parse)
workflow.add_node("node_tourism_meta_extract", node_tourism_meta_extract)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# 3. 指定入口节点
workflow.set_entry_point("node_entry")


# 4. 设置入口节点后的条件边（md / pdf / 表格 / 文本类 / 其他→END）
def after_entry_node(state: TourismImportGraphState):
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    elif state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    elif state.get("is_table_read_enabled"):
        return "node_tabular_parse"
    elif state.get("is_text_read_enabled"):
        return "node_text_parse"
    else:
        return END


workflow.add_conditional_edges("node_entry", after_entry_node, {
    "node_md_img": "node_md_img",
    "node_pdf_to_md": "node_pdf_to_md",
    "node_tabular_parse": "node_tabular_parse",
    "node_text_parse": "node_text_parse",
    END: END,
})

# 5. 设置静态边
# PDF 分支：pdf_to_md → md_img → document_split
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
# 文本类分支：text_parse → node_md_img → document_split（docx/html 内嵌图片经 enrich 增强；txt/json 无图时 node_md_img 安全早返回）
workflow.add_edge("node_text_parse", "node_md_img")
# 表格分支：tabular_parse 跳过切分，直接进入元数据抽取
workflow.add_edge("node_tabular_parse", "node_tourism_meta_extract")
# 汇合后统一走：元数据抽取 → 向量化 → 入库
workflow.add_edge("node_document_split", "node_tourism_meta_extract")
workflow.add_edge("node_tourism_meta_extract", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# 6. 编译图对象
kb_tourism_import_app = workflow.compile()


if __name__ == "__main__":
    import os
    import uuid

    logger.info("===== 开始执行文旅导入全流程测试 =====")
    test_state = create_default_state(
        task_id=f"test_tourism_import_{uuid.uuid4().hex[:8]}",
        local_file_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "doc", "兵马俑.pdf"),
        local_dir=os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "output"),
    )
    test_state["local_file_path"] = os.path.abspath(test_state["local_file_path"])
    test_state["local_dir"] = os.path.abspath(test_state["local_dir"])

    if not os.path.exists(test_state["local_file_path"]):
        logger.error(f"测试文件不存在：{test_state['local_file_path']}")
        logger.info("请将测试 PDF/MD/XLSX 放入 doc 目录下")
    else:
        try:
            logger.info(f"测试任务启动，文件路径：{test_state['local_file_path']}")
            final_state = None
            for step in kb_tourism_import_app.stream(test_state, stream_mode="values"):
                current_node = list(step.keys())[-1] if step else "未知"
                logger.info(f"节点执行完成：{current_node}")
                final_state = step

            if final_state:
                logger.info("-" * 80)
                logger.info("===== 文旅导入全流程测试成功 =====")
                chunks = final_state.get("chunks", [])
                logger.info(f"切分总数：{len(chunks)}")
                logger.info(f"主体名：{final_state.get('item_name', '')}")
                logger.info(f"元数据：{final_state.get('tourism_meta', {})}")
                has_embedding = all("dense_vector" in c and "sparse_vector" in c for c in chunks) if chunks else False
                logger.info(f"全部向量化完成：{'是' if has_embedding else '否'}")
                logger.info("-" * 80)
        except Exception as e:
            logger.exception("===== 文旅导入全流程测试失败 =====")
    logger.info("===== 文旅导入全流程测试结束 =====")
