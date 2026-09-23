"""
模型预热模块：在服务启动阶段一次性完成本地模型加载与首次推理。

背景（实测数据）：
BGE-M3 与 bge-reranker-large 均为懒加载单例，服务重启后的【第一次查询】需要额外承担
约 2.5s 的 CUDA 初始化 + 模型加载耗时，jieba 首次分词还要加载词典，Milvus 首次连接。这让"首字延迟"被无谓地拉长。

本模块把这些一次性成本全部前移到服务启动阶段，使首查直接进入热态。
预热失败不阻断服务启动（降级为旧的懒加载行为），仅记录告警。
"""
import time
from collections.abc import Callable

from app.shared.runtime.logger import logger, step_log

# 预热用的最小文本：只求触发真实推理路径，不追求语义
_DUMMY_QUERY = "预热"
_DUMMY_DOC = "预热文档"


@step_log("warmup_jieba")
def warmup_jieba() -> None:
    """预加载 jieba 词典（首次 lcut 会加载前缀词典，约 0.3~0.5s）。"""
    try:
        import jieba

        jieba.lcut(_DUMMY_QUERY)
        logger.debug("jieba 词典预热完成")
    except Exception as exc:
        logger.warning(f"jieba 预热失败，将在首次使用时加载：{exc}")

# 预热钩子注册表：依赖 rag/infra 的具体预热项（embedding/reranker/milvus/bm25）
# 由 app/api/bootstrap.py 在应用启动时注册，本模块保持对业务层零静态依赖（支撑分布式拆包）。
_WARMUP_HOOKS: list[tuple[str, Callable]] = []


def register_warmup_hook(name: str, fn: Callable) -> None:
    """注册一个预热钩子（按 name 去重，重复注册只保留首条）。"""
    if any(n == name for n, _ in _WARMUP_HOOKS):
        return
    _WARMUP_HOOKS.append((name, fn))

def warmup_all(enable_reranker: bool = True) -> dict[str, float]:
    """
    执行全部预热任务，返回各步骤耗时（毫秒）。

    Args:
        enable_reranker: 是否预热重排模型（仅查询服务需要，导入服务可关闭）。

    Returns:
        dict[str, float]: 预热项 → 耗时毫秒。
    """
    tasks: list[tuple[str, Callable]] = [("jieba", warmup_jieba), *_WARMUP_HOOKS]

    cost_map: dict[str, float] = {}
    start_ts = time.time()
    logger.info("===== 开始模型预热 =====")
    for name, task in tasks:
        task_start = time.time()
        try:
            task()
            cost_map[name] = int((time.time() - task_start) * 1000)
        except Exception as exc:
            # 预热失败不阻断启动：退化为懒加载，服务仍可用
            cost_map[name] = -1
            logger.warning(f"预热项[{name}]失败，将退化为懒加载：{exc}")
    total_ms = int((time.time() - start_ts) * 1000)
    cost_map["total"] = total_ms
    logger.info(f"===== 模型预热结束，总耗时={total_ms}ms，明细={cost_map} =====")
    return cost_map
