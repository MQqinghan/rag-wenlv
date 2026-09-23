"""
测试模块中的 02_test_query_maingreph 模块，负责承载对应场景的具体实现逻辑。
"""
import json

from app.process.query.agent.main_graph import query_app
from app.process.query.agent.state import create_query_default_state
from app.shared.runtime.logger import logger

logger.info("===== 开始测试 =====")

initial_state = create_query_default_state(session_id="test_001",
        original_query="华为P60怎么样?")
final_state = None

# 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息
for event in query_app.stream(initial_state):  # invoke   astream  ainvoke
    for key, value in event.items():
        logger.info(f"节点: {key}")
        final_state = value

# 格式化输出最终状态
logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}")

logger.info("图结构:")
# uv add grandalf
query_app.get_graph().print_ascii()

logger.info("===== 测试结束 =====")