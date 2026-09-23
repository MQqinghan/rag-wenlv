"""
知识库数据版本号同步服务：解决"导入服务与查询服务分属两个进程，导入后查询进程内存缓存不失效"的问题。

背景：
    import_server 与 query_server 是两个独立进程，导入完成后无法直接清空查询进程内的
    两件套内存缓存（BM25 索引 / LLM 结果缓存）。三个缓存各自原有的失效函数
    （invalidate_bm25_cache / clear_cache）只能作用于本进程。

机制（Mongo kv 广播）：
    - 导入侧：导入任务成功后调用 notify_knowledge_updated()，先清本进程缓存（兼容同进程
      部署形态），再把新版本号（时间戳）写入 Mongo 的 app_meta 集合广播。
    - 查询侧：每次查询入口先调用 maybe_refresh_query_caches()，内部按 CHECK_INTERVAL_SECONDS
      节流读取版本号，发现版本变化才真正刷新两件套缓存（读 Mongo 很便宜，刷新两件套较贵，
      节流是为了不让高频查询放大 Mongo 压力）。

降级策略：
    所有 Mongo 操作均 try/except 静默降级——Mongo 不可用时不阻断导入收尾、不阻断查询主链路，
    此时退化为 BM25/LLM 缓存自带的 600s TTL 兜底。
"""
import time
from collections.abc import Callable

from app.shared.runtime.logger import logger

# 版本号 kv 集合与文档键（连接配置复用 mongo_history_utils：MONGO_URL / MONGO_DB_NAME）
_META_COLLECTION = "app_meta"
_VERSION_DOC_KEY = "knowledge_version"

# 查询侧版本检查节流间隔（秒）：距上次检查不足该间隔时直接跳过 Mongo 读取
CHECK_INTERVAL_SECONDS = 5.0

# 查询侧节流状态（模块级变量）。单进程多请求线程共享：并发下最多多做一次重复检查/刷新，
# 各缓存失效函数本身幂等，无需加锁。
_last_checked_ts: float = 0.0
_last_seen_version: float | None = None

# 惰性获取的 app_meta collection（进程内缓存一次即可，避免每次查询重建集合对象）
_meta_collection = None


def _get_meta_collection():
    """
    惰性获取 app_meta collection；连接失败返回 None（降级为无广播能力，不抛异常）。

    复用 mongo_history_utils 的 HistoryMongoTool 单例：同一进程共用一条 Mongo 连接，
    避免重复建连；其内部已做预初始化 + 懒加载兜底。
    """
    global _meta_collection
    if _meta_collection is not None:
        return _meta_collection
    try:
        # 函数内导入：避免模块加载期拉起 Mongo 连接与重依赖链（对齐"降级不阻断启动"原则）
        from app.shared.clients.mongo_history_utils import get_history_mongo_tool

        mongo_tool = get_history_mongo_tool()
        _meta_collection = mongo_tool.db[_META_COLLECTION]
        return _meta_collection
    except Exception as e:
        logger.warning(f"[data_version] 获取 Mongo collection 失败，版本号广播降级停用：{e}")
        return None


def bump_data_version() -> bool:
    """
    写入新的知识库数据版本号（当前时间戳，upsert）。

    :return: 写入成功返回 True；Mongo 不可用等失败场景返回 False（调用方降级处理）。
    """
    coll = _get_meta_collection()
    if coll is None:
        return False
    try:
        new_version = time.time()
        coll.update_one(
            {"_id": _VERSION_DOC_KEY},
            {"$set": {"version": new_version}},
            upsert=True,
        )
        return True
    except Exception as e:
        logger.warning(f"[data_version] 知识库版本号写入失败（Mongo 不可用？）：{e}")
        return False


def get_data_version() -> float | None:
    """
    读取当前知识库数据版本号。

    :return: 版本号（时间戳浮点）；Mongo 不可用或尚无版本记录时返回 None。
    """
    coll = _get_meta_collection()
    if coll is None:
        return None
    try:
        doc = coll.find_one({"_id": _VERSION_DOC_KEY})
        if not doc:
            return None
        return float(doc.get("version", 0)) or None
    except Exception as e:
        logger.warning(f"[data_version] 知识库版本号读取失败（Mongo 不可用？）：{e}")
        return None


# 查询侧缓存失效钩子注册表：依赖 rag 层的具体失效函数（invalidate_bm25_cache）
# 由 app/api/bootstrap.py 在应用启动时注册，本模块保持对业务层零静态依赖（支撑分布式拆包）。
QUERY_CACHE_INVALIDATORS: list[tuple[str, Callable]] = []


def register_query_cache_invalidator(name: str, fn: Callable) -> None:
    """注册一个查询侧缓存失效钩子（按 name 去重）。"""
    if any(n == name for n, _ in QUERY_CACHE_INVALIDATORS):
        return
    QUERY_CACHE_INVALIDATORS.append((name, fn))


def _refresh_query_side_caches() -> None:
    """
    清空查询进程内的缓存。

    具体失效函数（BM25 索引，依赖 rag 层）由 app/api/bootstrap.py 在应用启动时注册；
    LLM 结果缓存为 shared 内部组件，此处直接调用。两项彼此独立：单项失败只记警告，不影响其余项。
    """
    from app.shared.runtime.llm_cache import clear_cache

    for name, invalidate_fn in (
        ("LLM结果缓存", clear_cache),
        *QUERY_CACHE_INVALIDATORS,
    ):
        try:
            invalidate_fn()
        except Exception as e:
            logger.warning(f"[data_version] 清空{name}失败：{e}")

def notify_knowledge_updated() -> None:
    """
    知识库导入成功后的通知入口（导入侧调用，仅成功路径调用）。

    两步：1) 先清本进程缓存——兼容"导入+查询同进程"部署形态；
          2) 再 bump 版本号广播——查询进程在下次查询时经节流比对后自行刷新缓存。
    本函数自身不抛异常，保证不影响导入任务收尾（任务状态已在此前置为 completed）。
    """
    try:
        _refresh_query_side_caches()
    except Exception as e:
        logger.warning(f"[data_version] 导入侧本进程缓存清理失败：{e}")
    if bump_data_version():
        logger.info("[data_version] 已广播知识库数据版本更新（各查询进程将在下次查询时刷新缓存）")
    else:
        logger.warning("[data_version] 版本号写入失败，其他进程将依赖缓存 TTL（600s）兜底失效")


def maybe_refresh_query_caches() -> None:
    """
    查询入口的节流版本检查（查询侧调用，放 invoke_query 最前面）。

    流程：距上次检查不足 CHECK_INTERVAL_SECONDS 直接返回 → 读版本号 →
    首次检查只记录基线不刷新（服务启动后缓存本就是空的，懒加载即最新）→
    版本与上次一致直接返回 → 版本变化则刷新两件套缓存并更新基线。
    本函数自身不抛异常，保证不影响查询主链路。
    """
    global _last_checked_ts, _last_seen_version
    try:
        now = time.time()
        if now - _last_checked_ts < CHECK_INTERVAL_SECONDS:
            return
        _last_checked_ts = now
        version = get_data_version()
        if version is None:
            return
        if _last_seen_version is None:
            # 进程首次检查：只记录基线。启动后缓存均为空，懒加载拉到的就是最新数据，无需刷新。
            _last_seen_version = version
            return
        if version == _last_seen_version:
            return
        _last_seen_version = version
        logger.info(f"[data_version] 检测到知识库数据版本更新 -> {version}，开始刷新查询侧缓存")
        _refresh_query_side_caches()
        logger.info("[data_version] 查询侧缓存刷新完成")
    except Exception as e:
        logger.warning(f"[data_version] 版本检查异常（忽略，不影响本次查询）：{e}")
