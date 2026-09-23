"""
文旅元数据抽取节点：从切片中抽取文旅元数据（主体名/内容类型/地区/文化主题/类别/专属字段），
回填到每个 chunk，并将主体向量化写入 Milvus tourism_entity 主体索引。
"""
from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.tourism_import.agent.state import TourismImportGraphState
from app.rag.tourism_import.tourism_meta_extract_service import extract_and_index_tourism_metadata


@node_log("node_tourism_meta_extract")
def node_tourism_meta_extract(state: TourismImportGraphState) -> TourismImportGraphState:
    """
    节点: 文旅元数据抽取 (node_tourism_meta_extract)
    为什么叫这个名字: 一次 LLM 调用抽取文旅全量元数据（分类+主体名+地区+文化主题+类别+专属字段）。
    """
    add_running_task(state["task_id"], "node_tourism_meta_extract")
    state = extract_and_index_tourism_metadata(state)
    add_done_task(state["task_id"], "node_tourism_meta_extract")
    return state


if __name__ == "__main__":
    logger.info("===== tourism node_tourism_meta_extract 单元测试 =====")
    try:
        mock_state = TourismImportGraphState({
            "task_id": "test_tourism_meta_node_001",
            "file_title": "西安景点资料",
            "local_file_path": "西安景点资料.pdf",
            "chunks": [
                {
                    "title": "景点简介",
                    "content": "兵马俑位于陕西西安，是秦始皇陵陪葬坑，国家5A级景区，门票120元，开放时间8:30-17:30。",
                },
                {
                    "title": "游览建议",
                    "content": "建议游览时长3小时，最佳季节春秋两季，可乘坐地铁9号线到达。",
                },
            ],
        })
        result_state = node_tourism_meta_extract(mock_state)
        logger.info(f"主体名：{result_state.get('item_name')}")
        logger.info(f"元数据：{result_state.get('tourism_meta')}")
        logger.info(f"切片数：{len(result_state.get('chunks', []))}")
        first_chunk = result_state.get("chunks", [{}])[0]
        logger.info(f"首切片 item_name：{first_chunk.get('item_name')}")
        logger.info(f"首切片 content_type：{first_chunk.get('content_type')}")
        logger.info(f"首切片 source_type：{first_chunk.get('source_type')}")
    except Exception as e:
        logger.error(f"节点测试失败：{e}", exc_info=True)
    logger.info("===== tourism node_tourism_meta_extract 测试结束 =====")
