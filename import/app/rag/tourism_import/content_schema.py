"""
文旅知识库内容类型 Schema：类型枚举 + 各类型专属字段定义。
给 LLM 元数据抽取对齐输出结构，同时是 Milvus 显式字段的映射源。
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TourismContentType(str, Enum):
    """文旅内容类型枚举（content_type 标量字段取值范围）。"""
    ATTRACTION_INFO = "景点信息"
    SCENIC_INTRO = "景区介绍"
    CULTURE_KNOWLEDGE = "文化知识介绍"
    TRAVEL_GUIDE = "游记攻略"
    OPERATION_MATERIAL = "推荐运营资料"
    REVIEW_SUMMARY = "游客评论摘要"
    FAQ = "常见问答"
    OPERATION_TABLE = "运营表格"   # Excel/CSV 来源的默认类型，可被前端指定覆盖
    LINE_RECOMMEND = "线路推荐"     # 行程/线路推荐
    HOTEL_INFO = "酒店信息"         # 酒店/住宿信息
    FOOD_RECOMMEND = "美食推荐"     # 美食/餐厅推荐
    TRAFFIC_GUIDE = "交通指南"      # 交通/出行方式指南


# ---- 各类型专属字段（存入 extra_meta JSON 字段）----

class AttractionExtra(BaseModel):
    """景点信息/景区介绍 专属字段"""
    attraction_name: Optional[str] = None      # 景点名称
    level: Optional[str] = None                # 景点等级（5A/4A/世界遗产等）
    ticket_price: Optional[str] = None          # 门票价格
    open_time: Optional[str] = None             # 开放时间
    best_season: Optional[str] = None           # 最佳游览季节
    visit_duration: Optional[str] = None        # 建议游览时长
    transportation: Optional[str] = None        # 交通方式
    nearby_facilities: Optional[str] = None     # 周边设施


class CultureKnowledgeExtra(BaseModel):
    """文化知识介绍 专属字段"""
    theme: Optional[str] = None                 # 知识主题（如：北欧文化、傣族泼水节）
    culture_category: Optional[str] = None      # 文化类别：非遗/民俗/历史/建筑/饮食/艺术…
    period: Optional[str] = None                # 历史时期
    summary: Optional[str] = None              # 核心内容摘要（≤200字）
    related_attractions: list[str] = []         # 关联景点


class TravelGuideExtra(BaseModel):
    """游记攻略 专属字段"""
    author: Optional[str] = None               # 作者
    route: Optional[str] = None                 # 行程路线（如：D1 西安→D2 华山）
    attractions: list[str] = []                # 涉及景点列表
    budget: Optional[str] = None               # 花费预算
    travel_time: Optional[str] = None          # 出行时间
    tips: Optional[str] = None                 # 实用贴士


class RouteExtra(BaseModel):
    """线路推荐 专属字段（旅行社线路/行程安排类资料）"""
    route_name: Optional[str] = None           # 线路名称（如：成都→九寨沟 3日游）
    days: Optional[str] = None                 # 行程天数（如：3天2晚）
    itinerary: Optional[str] = None            # 每日行程安排
    attractions: list[str] = []                # 途经景点列表
    transportation: Optional[str] = None       # 交通方式（大巴/高铁/飞机等）
    budget: Optional[str] = None               # 参考价格/人均预算
    tips: Optional[str] = None                 # 实用贴士


class HotelExtra(BaseModel):
    """酒店信息 专属字段（酒店/民宿/住宿推荐类资料）"""
    hotel_name: Optional[str] = None           # 酒店名称
    star_level: Optional[str] = None           # 星级（五星/四星/特色民宿等）
    price_range: Optional[str] = None          # 价格区间
    location: Optional[str] = None             # 位置/地址
    facilities: Optional[str] = None           # 设施服务（泳池/早餐/停车场等）
    nearby_attractions: list[str] = []         # 周边景点
    tips: Optional[str] = None                 # 实用贴士


class FoodExtra(BaseModel):
    """美食推荐 专属字段（餐厅/小吃/美食街推荐类资料）"""
    restaurant_name: Optional[str] = None      # 餐厅/店铺名称
    cuisine: Optional[str] = None              # 菜系/风味（川菜/粤菜/本地小吃等）
    signature_dishes: list[str] = []           # 招牌菜/必点菜
    price_range: Optional[str] = None          # 人均消费/价格区间
    location: Optional[str] = None             # 位置/地址
    opening_hours: Optional[str] = None        # 营业时间
    tips: Optional[str] = None                 # 实用贴士


class TrafficExtra(BaseModel):
    """交通指南 专属字段（机场/高铁/公共交通/自驾等出行指引）"""
    traffic_mode: Optional[str] = None         # 交通方式（飞机/高铁/地铁/自驾等）
    route_desc: Optional[str] = None           # 线路/路线描述
    duration: Optional[str] = None             # 所需时间/时长
    cost: Optional[str] = None                 # 费用
    schedule: Optional[str] = None             # 班次/发车时间表
    tips: Optional[str] = None                 # 实用贴士


# ---- 统一的抽取结果（LLM 输出对齐该结构）----

class TourismMetadata(BaseModel):
    """LLM 元数据抽取的统一输出结构，同时是对齐 Milvus 显式字段的映射源。"""
    item_name: str = Field(description="主体名：景点名/文化主题/游记标题，无法判断则空串")
    content_type: TourismContentType = Field(description="内容类型")
    region: str = Field(default="", description="所属地区：省/市或国家/民族文化范围")
    cultural_theme: str = Field(default="", description="文化主题，仅文化类内容填写")
    category: str = Field(default="", description="类别标签，如：自然风光/历史古迹/亲子/民俗文化")
    extra: dict = Field(default_factory=dict, description="类型专属字段，按 content_type 填充")


# 类型 → 专属字段模型 的映射（抽取结果校验用）
EXTRA_MODELS: dict[TourismContentType, type[BaseModel]] = {
    TourismContentType.ATTRACTION_INFO: AttractionExtra,
    TourismContentType.SCENIC_INTRO: AttractionExtra,
    TourismContentType.CULTURE_KNOWLEDGE: CultureKnowledgeExtra,
    TourismContentType.TRAVEL_GUIDE: TravelGuideExtra,
    TourismContentType.LINE_RECOMMEND: RouteExtra,
    TourismContentType.HOTEL_INFO: HotelExtra,
    TourismContentType.FOOD_RECOMMEND: FoodExtra,
    TourismContentType.TRAFFIC_GUIDE: TrafficExtra,
}


def normalize_extra(content_type: TourismContentType | str, extra: dict | None) -> dict:
    """
    按内容类型清洗 LLM 抽取的 extra 字段（键名白名单过滤）。

    背景：LLM 抽取时可能把 prompt 说明文字或类型名误当键（如
    {"仅根据 content_type 填对应类型的字段…": ""}、{"景点信息/景区介绍": "..."}），
    本函数在入库前把键收窄到该类型专属模型的字段集，脏键一律剔除。

    Args:
        content_type: 内容类型（枚举值或其中文字符串）。
        extra: LLM 抽取的原始 extra dict；为 None 或空时返回空 dict。

    Returns:
        dict: 清洗后的 extra；无专属模型覆盖的类型（运营表格/评论/FAQ 等）返回空 dict。
    """
    if not extra:
        return {}
    if isinstance(content_type, str):
        try:
            content_type = TourismContentType(content_type)
        except ValueError:
            return {}
    model = EXTRA_MODELS.get(content_type)
    if model is None:
        # 无专属字段模型的类型：一律清空（运营资料/评论摘要/FAQ/运营表格 等不承载专属字段）
        return {}
    allowed_keys = set(model.model_fields.keys())
    return {key: value for key, value in extra.items() if key in allowed_keys}


if __name__ == "__main__":
    # 单元测试：枚举 + 解析 + 降级兜底
    from app.shared.runtime.logger import logger

    logger.info("===== content_schema 单元测试 =====")

    # 测试1：正常 JSON 解析
    raw_json = '{"item_name": "兵马俑", "content_type": "景点信息", "region": "陕西西安", "cultural_theme": "", "category": "历史古迹", "extra": {"attraction_name": "秦始皇兵马俑", "level": "5A", "ticket_price": "120元"}}'
    meta = TourismMetadata.model_validate_json(raw_json)
    logger.info(f"测试1 item_name={meta.item_name}, type={meta.content_type.value}, region={meta.region}")
    assert meta.item_name == "兵马俑"
    assert meta.content_type == TourismContentType.ATTRACTION_INFO
    assert meta.extra.get("level") == "5A"

    # 测试2：文化知识类（纯文化资料，非景点）
    raw_culture = '{"item_name": "北欧文化", "content_type": "文化知识介绍", "region": "北欧", "cultural_theme": "北欧文化", "category": "民俗文化", "extra": {"theme": "北欧文化特点", "culture_category": "民俗"}}'
    meta2 = TourismMetadata.model_validate_json(raw_culture)
    logger.info(f"测试2 type={meta2.content_type.value}, theme={meta2.cultural_theme}")
    assert meta2.content_type == TourismContentType.CULTURE_KNOWLEDGE

    # 测试3：降级兜底（item_name 为空时不在此处兜底，由 service 层用 file_title 兜底）
    raw_empty = '{"item_name": "", "content_type": "运营表格", "region": "", "cultural_theme": "", "category": "", "extra": {}}'
    meta3 = TourismMetadata.model_validate_json(raw_empty)
    logger.info(f"测试3 降级 item_name=[{meta3.item_name}], type={meta3.content_type.value}")
    assert meta3.content_type == TourismContentType.OPERATION_TABLE

    # 测试4：JSON Schema 生成（前端/文档用）
    schema = TourismMetadata.model_json_schema()
    logger.info(f"测试4 JSON Schema 含字段: {list(schema.get('properties', {}).keys())}")

    # 测试5：新增 4 类枚举可解析
    for value in ("线路推荐", "酒店信息", "美食推荐", "交通指南"):
        assert TourismContentType(value).value == value
    logger.info(f"测试5 内容类型枚举共 {len(TourismContentType)} 类: {[t.value for t in TourismContentType]}")

    # 测试6：normalize_extra 脏键清洗（LLM 误把说明文字/类型名当键 → 应被剔除）
    dirty = {
        "仅根据 content_type 填对应类型的字段，原文没有的字段填 null：": "",
        "景点信息/景区介绍": "attraction_name, level, ...",
        "attraction_name": "兵马俑",
        "level": "5A",
        "not_a_field": "x",
    }
    cleaned = normalize_extra(TourismContentType.ATTRACTION_INFO, dirty)
    logger.info(f"测试6 清洗后 extra={cleaned}")
    assert cleaned == {"attraction_name": "兵马俑", "level": "5A"}

    # 测试7：无专属模型类型（运营表格）extra 一律清空
    cleaned_table = normalize_extra(TourismContentType.OPERATION_TABLE, {"a": "b", "level": "5A"})
    logger.info(f"测试7 运营表格清洗后 extra={cleaned_table}")
    assert cleaned_table == {}

    # 测试8：字符串形式 content_type 也可清洗；None/空 dict 返回空
    assert normalize_extra("酒店信息", {"hotel_name": "A", "脏键": 1}) == {"hotel_name": "A"}
    assert normalize_extra("酒店信息", None) == {}
    assert normalize_extra("非法类型", {"a": 1}) == {}
    logger.info("测试8 normalize_extra 边界通过")

    logger.info("===== content_schema 测试通过 =====")
