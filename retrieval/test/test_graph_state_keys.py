"""
LangGraph state 键声明一致性静态扫描（秒级，零 LLM、零 Milvus）。

为什么要这个测试：
    LangGraph 对未在 TypedDict 中声明的 state 键会**静默丢弃写入**，不报错、不告警，
    表现为「代码明明写了却完全不生效」，排查成本极高。本项目已踩过两次：
      - 历史上曾出现节点写入键未在 TypedDict 声明，导致整条链路静默失效
    这类问题都是靠人工读评测结果才发现的，本测试把它变成秒级自动拦截。

运行：
    python test/test_graph_state_keys.py

扫描逻辑（纯 AST，不 import 任何业务模块，因此不加载模型、不连 Milvus）：
    1) 解析两个图的 state 定义文件，收集所有 TypedDict 声明的键（含继承链）
    2) 解析所有 node 模块，收集节点写入 state 的键：
       - `return {"key": ...}` 的字典字面量键
       - `state["key"] = ...` 的下标赋值键
    3) 报告「写入了但未声明」的键 —— 即潜在的静默丢弃点
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STATE_FILES = [
    "app/process/unified_query/agent/state.py",
]

# 查询链路扫描范围：只覆盖会写「查询图 state」的模块。
# 注意：导入侧（app/rag/*_import/、split_service 等）用的是另一套导入图 state，
# 混入会大量误报，故显式不纳入。
QUERY_SCAN_DIRS = [
    "app/process/unified_query/agent/nodes",
    "app/rag/tourism_query",
]
QUERY_SCAN_FILES = [
    "app/rag/common/intent_route_service.py",
    "app/rag/common/chitchat_answer_service.py",
    "app/rag/common/clarify_answer_service.py",
    "app/rag/common/datetime_answer_service.py",
]

# 白名单：确认是动态/临时键、不进 state 通道的（新增前请先确认真的不需要声明）
ALLOW_UNDECLARED = {
    "test",       # 节点内 mock 用
}

# 关键 state 键兜底断言：即使静态扫描因写法特殊（如中间变量 result[...] 再回填）
# 漏检，这些键一旦从 TypedDict 消失也必须立刻失败。均为本项目真实踩坑或主干字段。
CRITICAL_KEYS = (
    "answer", "domain", "rewritten_query", "history",
    "is_plan",          # 行程分支开关
    "route_info",       # B2/B3 域判定唯一数据源
    "reranked_docs", "rrf_chunks",
    "web_search_docs",
    "tool_weather", "tool_route",
    "item_names",
)

# 已知缺失但不阻断：
#   item_names —— attraction_confirm_service.py:164 写入、:182 同函数内读取，
#   下游节点不读该键，故当前无功能影响；但未声明意味着它**不会跨节点传递**，
#   一旦将来下游要按 item_names 做定向检索就会静默失效。
#   是否补 TypedDict 声明属主链行为变更，待主人决定后再移除本豁免。
KNOWN_MISSING = {"item_names"}


def collect_declared_keys() -> tuple[set[str], dict[str, str]]:
    """从 state 定义文件收集所有 TypedDict 声明的键（含基类继承）。"""
    classes: dict[str, set[str]] = {}
    bases: dict[str, list[str]] = {}

    for rel in STATE_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            keys = set()
            for stmt in node.body:
                # 形如 `key: type` 的注解声明
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    keys.add(stmt.target.id)
                # 形如 `key: type = default`（部分 TypedDict 写法）
                elif isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, ast.Name):
                            keys.add(t.id)
            classes[node.name] = keys
            bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]

    # 展开继承链
    def expand(name: str, seen: set[str] | None = None) -> set[str]:
        seen = seen or set()
        if name in seen or name not in classes:
            return set()
        seen.add(name)
        out = set(classes[name])
        for b in bases.get(name, []):
            out |= expand(b, seen)
        return out

    declared: set[str] = set()
    owner: dict[str, str] = {}
    for name in classes:
        for k in expand(name):
            declared.add(k)
            owner.setdefault(k, name)
    return declared, owner


def collect_written_keys() -> dict[str, set[str]]:
    """扫描 node 模块，收集写入 state 的键。返回 {文件路径: {键}}。"""
    written: dict[str, set[str]] = {}
    targets: list[Path] = []
    for rel in QUERY_SCAN_DIRS:
        d = ROOT / rel
        if d.exists():
            targets += sorted(d.glob("*.py"))
    for rel in QUERY_SCAN_FILES:
        f = ROOT / rel
        if f.exists():
            targets.append(f)

    for path in targets:
        if path.name.startswith("_") or "mock" in path.name.lower():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as e:
            written[str(path.relative_to(ROOT))] = {f"<语法错误 {e}>"}
            continue

        # 只有 node_* 模块的 return 才是「state 增量」；
        # 服务层函数的 return 多为业务字典（检索结果 doc、工具简报等），
        # 若一并计入会把 content/score/title 之类误报成 state 键。
        is_node_module = path.name.startswith("node_")

        keys: set[str] = set()
        for node in ast.walk(tree):
            # return {"key": ...}
            if is_node_module and isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                for k in node.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        keys.add(k.value)
        # state["key"] = ...
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and t.value.id in ("state", "st")
                        and isinstance(t.slice, ast.Constant)
                        and isinstance(t.slice.value, str)):
                    keys.add(t.slice.value)
        # state.update({"key": ...})
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("state", "st")):
            for a in node.args:
                if isinstance(a, ast.Dict):
                    for k in a.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            keys.add(k.value)
        if keys:
            written[str(path.relative_to(ROOT))] = keys
    return written


def main() -> None:
    declared, owner = collect_declared_keys()
    written = collect_written_keys()

    print("== graph state 键声明一致性扫描 ==")
    print(f"  声明的 state 键：{len(declared)} 个")
    print(f"  扫描 node 文件：{len(written)} 个")

    undeclared: list[tuple[str, str]] = []
    for path, keys in sorted(written.items()):
        for k in sorted(keys):
            if k.startswith("<"):  # 语法错误占位
                undeclared.append((path, k))
            elif k not in declared and k not in ALLOW_UNDECLARED:
                undeclared.append((path, k))

    missing_critical = [k for k in CRITICAL_KEYS if k not in declared and k not in KNOWN_MISSING]
    known = [k for k in CRITICAL_KEYS if k in KNOWN_MISSING and k not in declared]
    for k in known:
        print(f"  ⚠️  已知缺失（暂不阻断）：{k} —— 不会跨节点传递，见文件内 KNOWN_MISSING 注释")

    results = [
        ("state 定义可解析", len(declared) > 0, f"{len(declared)} 个键"),
        ("node 写入键均已声明", not undeclared,
         "全部已声明" if not undeclared else f"{len(undeclared)} 个未声明"),
        ("关键 state 键未丢失", not missing_critical,
         f"{len(CRITICAL_KEYS)} 个关键键齐全" if not missing_critical
         else f"丢失 {missing_critical}"),
    ]

    for name, passed, info in results:
        print(f"  {'✅' if passed else '❌'} {name}：{info}")

    if undeclared:
        print("\n  未声明的 state 键（LangGraph 会静默丢弃，务必补进 TypedDict）：")
        for path, k in undeclared:
            print(f"    - {k}  ←  {path}")
    if missing_critical:
        print("\n  关键 state 键从 TypedDict 中丢失（会静默丢弃）：")
        for k in missing_critical:
            print(f"    - {k}")
    if undeclared or missing_critical:
        print("\n  修复方式：在对应 state.py 的 TypedDict 中补 `键名: 类型` 声明。")
        raise SystemExit(1)

    print("全部通过 ✅")


if __name__ == "__main__":
    main()
