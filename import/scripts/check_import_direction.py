#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 app/ 内跨包 import 方向，强制依赖单向：
       shared <- infra <- rag <- process <- api
   同包内任意；禁止高层 import 低层（反向依赖）。
   用法：
       python scripts/check_import_direction.py [--root .] [--no-fail]
   退出码：发现违规且非 --no-fail 时返回 1（供 CI/pre-commit 拦截）。
"""
import argparse
import os
import re
import sys

# 包级依赖顺序：数值越小越底层，允许被更高层依赖
PKG_ORDER = {"shared": 0, "infra": 1, "rag": 2, "process": 3, "api": 4}

IMPORT_RE = re.compile(
    r'^\s*(?:from\s+app\.(\w+)(?:\.\w+)*\s+import|import\s+app\.(\w+)(?:\.\w+)*)'
)


def scan(root: str):
    violations = []
    app_dir = os.path.join(root, "app")
    for dirpath, _, fnames in os.walk(app_dir):
        for fn in fnames:
            if not fn.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fn)
            rel = os.path.relpath(fpath, root).replace(os.sep, "/")
            parts = rel.split("/")
            src_pkg = parts[1] if rel.startswith("app/") and len(parts) > 1 else None
            if src_pkg not in PKG_ORDER:
                continue
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                for ln, line in enumerate(f, 1):
                    m = IMPORT_RE.match(line)
                    if not m:
                        continue
                    dep = m.group(1) or m.group(2)
                    if dep == src_pkg or dep not in PKG_ORDER:
                        continue
                    # 允许 src 依赖更底层（order 小）或同层；违规=依赖了更高层
                    if PKG_ORDER[dep] > PKG_ORDER[src_pkg]:
                        violations.append((rel, ln, line.strip(), src_pkg, dep))
    return violations


def main():
    ap = argparse.ArgumentParser(description="app 跨包 import 方向检查")
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-fail", action="store_true", help="只报告不返回非零退出码")
    args = ap.parse_args()

    violations = scan(args.root)
    if violations:
        print(f"[FAIL] 发现 {len(violations)} 处反向 import 依赖（违反 api→process→rag/infra→shared 方向）：")
        for rel, ln, code, src, dep in violations:
            print(f"  {rel}:{ln}  {src} 反向依赖 {dep}  | {code}")
    else:
        print("[OK] 未发现反向 import 依赖（依赖方向合规）")

    if violations and not args.no_fail:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
