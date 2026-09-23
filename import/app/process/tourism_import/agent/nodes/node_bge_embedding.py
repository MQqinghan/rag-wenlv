"""
文旅导入流程节点：BGE-M3 向量化（稠密+稀疏）。
前缀拼接："主体:{item_name},类型:{content_type},内容:{content}"。
"""
import os
from dotenv import load_dotenv

from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.tourism_import.agent.state import TourismImportGraphState
from app.rag.tourism_import.embedding_service import generate_chunk_embeddings


@node_log("node_bge_embedding")
def node_bge_embedding(state: TourismImportGraphState) -> TourismImportGraphState:
    """
    作用: chunks - chunk - 生成稠密和稀疏向量
    细节: 1. 批量生成 embedding 2. 语义增强 主体+类型+content 3. 批量处理(异常处理)
    """
    add_running_task(state["task_id"], "node_bge_embedding")
    state = generate_chunk_embeddings(state)
    add_done_task(state["task_id"], "node_bge_embedding")
    return state


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    test_state = TourismImportGraphState({
        "task_id": "test_tourism_embed_node_001",
        "chunks": [
            {
                "content": "兵马俑位于陕西西安，是秦始皇陵陪葬坑，5A景区，门票120元。",
                "title": "景点简介",
                "item_name": "兵马俑",
                "content_type": "景点信息",
                "file_title": "西安景点资料",
            },
            {
                "content": "北欧文化以极简、自然、平等为核心特点。",
                "title": "文化特点",
                "item_name": "北欧文化",
                "content_type": "文化知识介绍",
                "file_title": "北欧文化资料",
            },
        ],
    })
    logger.info("=== tourism BGE-M3 向量化节点测试启动 ===")
    try:
        result_state = node_bge_embedding(test_state)
        result_chunks = result_state.get("chunks", [])
        logger.info(f"待处理 2 | 实际处理 {len(result_chunks)}")
        if result_chunks:
            logger.info(f"首切片 dense 维度：{len(result_chunks[0].get('dense_vector', []))}")
            logger.info(f"首切片 sparse 键数：{len(result_chunks[0].get('sparse_vector', {}))}")
    except Exception as e:
        logger.error(f"向量化节点测试失败（检查 BGE-M3 路径/显存）: {e}", exc_info=True)
