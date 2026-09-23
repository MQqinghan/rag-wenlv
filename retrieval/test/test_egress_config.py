# -*- coding: utf-8 -*-
"""
A1 外发总开关 IMPORT_EGRESS_MODE 档位解析单测（T2 验收）。

运行方式（与 test/ 下脚本式用例一致，不依赖 pytest）：
  python test/test_egress_config.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.shared.config import egress_config as ec  # noqa: E402


def _with_env(value, fn):
    """临时改写环境变量后执行 fn，结束后恢复原值。"""
    old = os.environ.get(ec.IMPORT_EGRESS_MODE_ENV)
    if value is None:
        os.environ.pop(ec.IMPORT_EGRESS_MODE_ENV, None)
    else:
        os.environ[ec.IMPORT_EGRESS_MODE_ENV] = value
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop(ec.IMPORT_EGRESS_MODE_ENV, None)
        else:
            os.environ[ec.IMPORT_EGRESS_MODE_ENV] = old


def main() -> None:
    # 1) 环境缺省 → 回落默认 mask（已拍板）
    assert _with_env(None, ec.import_egress_mode) == "mask", "缺省应回落 mask"
    # 2) 三档透传 + 大小写归一
    assert _with_env("off", ec.import_egress_mode) == "off"
    assert _with_env("FULL", ec.import_egress_mode) == "full"
    assert _with_env("mask", ec.import_egress_mode) == "mask"
    # 3) 非法/空值 → 回落默认 mask（宁保守勿泄露）
    assert _with_env("everything", ec.import_egress_mode) == "mask"
    assert _with_env("", ec.import_egress_mode) == "mask"
    # 4) is_egress_enabled：off 关闭，mask/full 开启
    assert _with_env("off", ec.is_egress_enabled) is False
    assert _with_env("mask", ec.is_egress_enabled) is True
    assert _with_env("full", ec.is_egress_enabled) is True
    # 5) 当前 .env 实际档位 = mask
    assert ec.import_egress_mode() == "mask", f"当前 .env 档位应为 mask，实际 {ec.import_egress_mode()}"
    print("egress_config 单测全部通过 ✅（default=off→mask 合法档/非法回落/开关判定/实配=mask）")


if __name__ == "__main__":
    main()
