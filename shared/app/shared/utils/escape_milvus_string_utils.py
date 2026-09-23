"""
Milvus 字符串转义工具模块
用于对 Milvus 布尔表达式中的字符串进行安全转义，防止注入攻击。
"""


def escape_milvus_string(value: str) -> str:
    """
    转义 Milvus 过滤表达式中的字符串值。
    处理规则：
    1. 转义反斜杠 \\ -> \\\\
    2. 转义单引号 ' -> \\'
    3. 转义双引号 " -> \\\"
    4. 最终结果用单引号包裹

    :param value: 需要转义的原始字符串
    :return: 转义后的字符串（已用单引号包裹）
    """
    if value is None:
        return "''"
    # 先转义反斜杠，再转义单引号和双引号
    escaped = str(value)
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace('"', '\\"')
    return f"'{escaped}'"
