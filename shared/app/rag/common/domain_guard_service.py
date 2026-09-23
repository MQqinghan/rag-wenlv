"""
领域误导预检服务（兼容入口）：当前系统仅文旅一个知识域，无领域误导风险。

inspect_import_guard 保留为兼容入口，恒返回可继续，供导入服务既有调用复用
（如后续扩展其它知识域，再恢复同名 / 特征词预检逻辑）。
"""

DOMAIN_TOURISM = "tourism"
_VALID_DOMAINS = (DOMAIN_TOURISM,)
DOMAIN_CN = {DOMAIN_TOURISM: "文旅"}


def inspect_import_guard(domain: str, stem: str) -> tuple[bool, str]:
    """
    领域误导预检主入口（单域模式）。

    Args:
        domain: 用户选择的导入领域（当前仅 tourism）。
        stem: 去扩展名文件名（将作为 Milvus file_title）。

    Returns:
        (ok, reason): 单域下恒为 (True, "")。
    """
    return True, ""
