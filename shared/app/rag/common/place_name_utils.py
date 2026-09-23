"""
地名归一化工具：行政区划后缀处理。

背景：LLM 解析出的出发地/目的地/城市，"成都" 与 "成都市" 两种形态随机出现。
下游判定若按字面比较，就会被这种形态差异误导——
- 素材闸门（itinerary_service）：正文写"成都"、解析出"成都市" → 零命中 → 误杀拒答；
- 路线工具（route_tool_service）："成都→成都市" 同城 → 高德返回 0 公里/0 分钟/0 元。
"""
import re

# 末尾的行政区划后缀；长的排前面，保证"自治区"先于"区"被匹配
_ADMIN_SUFFIX_PATTERN = re.compile(r"(?:自治区|自治州|自治县|地区|省|市|县|区|盟|旗)+$")
# 地名内部的分隔符，用于把"西安 兵马俑"这类复合值拆成片段
_SPLIT_PATTERN = re.compile(r"[\s，,、]+")


def strip_admin_suffix(name: str) -> str:
    """
    去掉地名末尾的行政区划后缀。

    Args:
        name: 原始地名，如"成都市""内蒙古自治区"。

    Returns:
        str: 去后缀结果，如"成都""内蒙古"；全被吃光（如"市区"）时返回原值。
    """
    text = str(name or "").strip()
    stripped = _ADMIN_SUFFIX_PATTERN.sub("", text)
    return stripped or text


def expand_place_tokens(*names: str) -> list[str]:
    """
    把若干地名扩展成判定用 token 列表：原样 + 去后缀变体，去重保序。

    Args:
        *names: 任意个地名字段（目的地、所属城市等）。

    Returns:
        list[str]: 判定 token 列表；长度 <2 的片段丢弃（过短易误命中）。
    """
    tokens: list[str] = []
    for name in names:
        for raw in _SPLIT_PATTERN.split(str(name or "")):
            if len(raw) < 2:
                continue
            for token in (raw, strip_admin_suffix(raw)):
                if len(token) >= 2 and token not in tokens:
                    tokens.append(token)
    return tokens
