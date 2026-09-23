"""
文旅导入流程节点：带向量的 chunks 写入 Milvus tourism_chunks 集合。
删旧仅按 file_title（同文件重导幂等覆盖），不按 item_name 删（同主体多文件共存）。
"""


from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.tourism_import.agent.state import TourismImportGraphState
from app.rag.tourism_import.index_service import index_chunks


@node_log("node_import_milvus")
def node_import_milvus(state: TourismImportGraphState) -> TourismImportGraphState:
    """
    作用: chunks 存到 Milvus (tourism_chunks)
    步骤: 校验 → 建集合 → 按 file_title 删旧（同文件重导幂等） → 插入新
    """
    add_running_task(state["task_id"], "node_import_milvus")
    state = index_chunks(state)
    add_done_task(state["task_id"], "node_import_milvus")
    return state


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    dim = 1024
    test_state = {
        "task_id": "test_tourism_milvus_node_001",
        "item_name": "兵马俑",
        "file_title": "西安景点资料",
        "chunks": [
            {
                "content": "兵马俑位于陕西西安，5A景区，门票120元。",
                "title": "景点简介",
                "item_name": "兵马俑",
                "content_type": "景点信息",
                "region": "陕西西安",
                "cultural_theme": "",
                "category": "历史古迹",
                "source_type": "pdf",
                "extra_meta": {"level": "5A"},
                "parent_title": "西安景点资料",
                "part": 1,
                "file_title": "西安景点资料",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8},
            }
        ],
    }
    logger.info("=== tourism Milvus 导入节点测试 ===")
    if not os.getenv("MILVUS_URL"):
        logger.warning("未设置 MILVUS_URL，跳过测试")
    else:
        try:
            result_state = node_import_milvus(test_state)
            logger.info(f"入库完成，chunks 数：{len(result_state.get('chunks', []))}")
        except Exception as e:
            logger.error(f"入库测试失败（检查 Milvus 连接）: {e}")
