"""
联网检索服务模块，负责通过 DashScope MCP 执行 WebSearch 并解析结果。

重要说明：
OpenAI Agents SDK 的 MCPServerStreamableHttp 在检测到 mcp>=2 时，
会使用 mode='auto' 发送 server/discover 握手请求（MCP v2 新协议）。
DashScope 百炼 MCP 服务端不支持 server/discover，返回 HTTP 500。
因此这里绕过 SDK，直接使用 mcp 库的 Client(mode='legacy') 发送传统 initialize 握手。
"""
import asyncio
import json

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from app.shared.config.bailian_mcp_config import mcp_config
from app.shared.config.common import env_float
from app.shared.runtime.logger import logger, step_log

DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key

# 联网搜索超时熔断阈值（秒）：联网是"锦上添花"的补充路，本地知识库才是主链路。
# 超过该时长直接放弃联网结果、降级为空列表，避免拖慢整个查询的关键路径。
WEB_SEARCH_TIMEOUT: float = env_float("WEB_SEARCH_TIMEOUT", default=2.5)

# ---- 联网调用埋点（T15：先埋点再优化；百炼 web search 29 元/千次）----
# 仅统计「真实发起」的联网调用；被 source_policy 门禁跳过的不会进此计数（另有图层埋点日志）。
_WEB_METRICS: dict = {"calls": 0, "total_pages": 0, "empty_calls": 0, "by_policy": {}}


def _record_web_call(state: dict, n_pages: int) -> None:
    """记录一次真实联网调用（含 policy 维度），供成本量化与门禁收益评估。"""
    policy = ((state.get("route_info") or {}).get("source_policy")) or "unknown"
    _WEB_METRICS["calls"] += 1
    _WEB_METRICS["total_pages"] += n_pages
    if n_pages == 0:
        _WEB_METRICS["empty_calls"] += 1
    _WEB_METRICS["by_policy"][policy] = _WEB_METRICS["by_policy"].get(policy, 0) + 1
    q = (state.get("rewritten_query") or "")[:40]
    logger.info(f"[埋点][web_search] policy={policy} hits={n_pages} query={q!r}")


def web_search_metrics() -> dict:
    """联网调用埋点快照（进程内累计），供成本量化与门禁收益评估。"""
    return {
        "calls": _WEB_METRICS["calls"],
        "total_pages": _WEB_METRICS["total_pages"],
        "empty_calls": _WEB_METRICS["empty_calls"],
        "by_policy": dict(_WEB_METRICS["by_policy"]),
    }


@step_log("validate_web_search_inputs")
def validate_web_search_inputs(state: dict) -> str:
    """
    校验联网检索所需的查询文本。

    Args:
        state: 查询图当前状态，需至少包含 `rewritten_query`。

    Returns:
        str: 已校验通过的改写查询文本。
    """
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error("rewritten_query不能为空!")
        raise ValueError("rewritten_query不能为空!")
    return rewritten_query


async def search_web_documents_async(rewritten_query: str, count: int = 5):
    """
    通过 DashScope MCP 异步执行一次联网搜索。

    使用 mcp.Client(mode='legacy') 发送传统 initialize 握手，
    绕过 SDK 的 mode='auto'（server/discover）避免 DashScope 500 错误。

    Args:
        rewritten_query: 改写后的查询文本。
        count: 搜索返回的最大结果数。
    """
    # 1. 创建带认证头的 httpx2 客户端
    http_client = httpx2.AsyncClient(
        follow_redirects=True,
        headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
        timeout=httpx2.Timeout(30.0, read=300.0),
    )
    # 2. 创建 Streamable HTTP 传输层
    transport = streamable_http_client(
        DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP,
        http_client=http_client,
    )
    # 3. 使用 mode='legacy' 创建 MCP 客户端，发送传统 initialize 握手
    client = Client(transport, mode="legacy")
    async with client:
        # 4. 调用百炼 WebSearch 工具
        return await client.call_tool(
            "bailian_web_search",
            {"query": rewritten_query, "count": count},
        )


@step_log("search_web_documents")
def search_web_documents(state: dict, count: int = 10) -> list[dict]:
    """
    执行联网检索并将 MCP 结果解析为页面列表。

    超时熔断：联网是补充路，超时直接降级为空结果，绝不让慢网拖垮主链路。

    Args:
        state: 查询图当前状态。
        count: 搜索返回的最大结果数。

    Returns:
        list[dict]: 联网搜索得到的页面结果列表；超时/失败时为空列表。
    """
    # 先校验改写问题，再把 MCP 原始结果解析成网页列表。
    rewritten_query = validate_web_search_inputs(state)
    try:
        mcp_result = asyncio.run(
            asyncio.wait_for(
                search_web_documents_async(rewritten_query, count=count),
                timeout=WEB_SEARCH_TIMEOUT,
            )
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(f"联网搜索超时（>{WEB_SEARCH_TIMEOUT}s），降级为空结果，不影响本地检索")
        _record_web_call(state, 0)
        return []
    text_dict = json.loads(mcp_result.content[0].text)
    pages = text_dict.get("pages", [])
    _record_web_call(state, len(pages))
    return pages
