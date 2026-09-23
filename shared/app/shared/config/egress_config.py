# -*- coding: utf-8 -*-
"""
导入侧外发（Egress）配置：`IMPORT_EGRESS_MODE` 三档读取与判定。

A1 配置分区（2026-09-09）落地；A2 `egress_gateway` 出口收敛、A3 脱敏/白名单、
A4 外发审计将围绕本模块扩展。

档位语义（见 docs/模块拆分与意图路由升级规划.md · A1）：
- off  ：一键关停所有外发 —— VL 调用跳过或显式报错，
         导入仍可完成（质量降级，零泄露）；
- mask ：脱敏后外发（默认档，脱敏口径见 A3）；
- full ：原样外发，仅用于已确认公开内容的批次。

读取策略：缺失或非法值一律回落默认档 mask（宁保守勿泄露），不抛异常；
非法值不阻断服务，调用方如需告警可自行比对 `raw` 与解析结果。
"""
from __future__ import annotations

from app.shared.config.common import env_str

#: 环境变量名（导入侧外发总开关）
IMPORT_EGRESS_MODE_ENV = "IMPORT_EGRESS_MODE"
#: 合法档位
EGRESS_MODES: tuple[str, ...] = ("off", "mask", "full")
#: 默认档（已拍板：mask）
EGRESS_MODE_DEFAULT = "mask"


def import_egress_mode() -> str:
    """
    读取导入侧外发档位。

    Returns:
        str: "off" / "mask" / "full" 之一；环境变量缺失或非法时回落默认 "mask"。
    """
    raw = env_str(IMPORT_EGRESS_MODE_ENV, EGRESS_MODE_DEFAULT)
    mode = raw.strip().lower()
    return mode if mode in EGRESS_MODES else EGRESS_MODE_DEFAULT


def is_egress_enabled() -> bool:
    """
    外发是否可用（off 关闭；mask/full 开启）。

    A2 egress_gateway 的总开关判定统一走这里，业务代码不直接读环境变量。
    """
    return import_egress_mode() != "off"
