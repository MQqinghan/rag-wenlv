"""
行程规划（结构化 JSON 行程）数据契约。

由 helloagents-trip-planner 的 backend/app/models/schemas.py 迁移改编：
- 去除 hello-agents 依赖，仅依赖 pydantic；
- 保留温度解析 field_validator；
- Location 增加 from_text 静态解析（兼容高德 "lng,lat" 字符串）。
"""
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "TripRequest",
    "Location",
    "RouteInfo",
    "Attraction",
    "Meal",
    "Hotel",
    "DayPlan",
    "WeatherInfo",
    "Budget",
    "TripPlan",
    "TripPlanResponse",
    "TripPlanNLRequest",
    "TripPlanErrorResponse",
]


class TripRequest(BaseModel):
    """旅行规划请求（与旧工程 TripRequest 字段保持 1:1）。"""

    city: str = Field(..., description="目的地城市", example="北京")
    start_date: str = Field(..., description="开始日期 YYYY-MM-DD", example="2026-09-20")
    end_date: str = Field(..., description="结束日期 YYYY-MM-DD", example="2026-09-22")
    travel_days: int = Field(..., ge=1, le=30, description="旅行天数", example=3)
    transportation: str = Field(..., description="交通方式", example="公共交通")
    accommodation: str = Field(..., description="住宿偏好", example="经济型酒店")
    preferences: List[str] = Field(default_factory=list, description="旅行偏好标签", example=["历史文化", "美食"])
    free_text_input: Optional[str] = Field(default="", description="额外要求", example="希望多安排一些博物馆")
    origin: Optional[str] = Field(default="", description="出发地城市（可选）；填写后行程将包含往返交通段", example="成都")
    budget: Optional[int] = Field(default=None, description="总预算（元，可选）；填写后行程规划将做预算校验与超支提示", example=5000)


class TripPlanNLRequest(BaseModel):
    """自然语言行程规划请求：从一句话中抽取 TripRequest 槽位。"""

    query: str = Field(..., description="用户的自然语言出行需求", example="帮我规划北京3天行程，9月20到22号，住经济型酒店，喜欢历史文化")
    session_id: Optional[str] = Field(default=None, description="会话ID，为空时自动生成")


class Location(BaseModel):
    """地理位置。"""

    longitude: float = Field(..., description="经度")
    latitude: float = Field(..., description="纬度")

    @staticmethod
    def from_text(loc_text: Optional[str]) -> Optional["Location"]:
        """解析高德 "lng,lat" 字符串；失败返回 None。"""
        if not loc_text or "," not in str(loc_text):
            return None
        try:
            lng_s, lat_s = str(loc_text).split(",", 1)
            return Location(longitude=float(lng_s), latitude=float(lat_s))
        except (ValueError, TypeError):
            return None


class Attraction(BaseModel):
    """景点信息。"""

    name: str = Field(..., description="景点名称")
    address: str = Field(..., description="地址")
    location: Optional[Location] = Field(default=None, description="经纬度坐标")
    visit_duration: int = Field(default=120, description="建议游览时间(分钟)")
    description: str = Field(default="", description="景点描述")
    category: Optional[str] = Field(default="景点", description="景点类别")
    rating: Optional[float] = Field(default=None, description="评分")
    photos: Optional[List[str]] = Field(default_factory=list, description="景点图片URL列表")
    poi_id: Optional[str] = Field(default="", description="高德 POI ID")
    image_url: Optional[str] = Field(default=None, description="图片URL")
    ticket_price: Optional[int] = Field(default=0, description="门票价格(元)")


class Meal(BaseModel):
    """餐饮信息。"""

    type: str = Field(..., description="餐饮类型: breakfast/lunch/dinner/snack")
    name: str = Field(..., description="餐饮名称")
    address: Optional[str] = Field(default=None, description="地址")
    location: Optional[Location] = Field(default=None, description="经纬度坐标")
    description: Optional[str] = Field(default=None, description="描述")
    estimated_cost: int = Field(default=0, description="预估费用(元)")


class Hotel(BaseModel):
    """酒店信息。"""

    name: str = Field(..., description="酒店名称")
    address: str = Field(default="", description="酒店地址")
    location: Optional[Location] = Field(default=None, description="酒店位置")
    price_range: str = Field(default="", description="价格范围")
    rating: str = Field(default="", description="评分")
    distance: str = Field(default="", description="距离景点距离")
    type: str = Field(default="", description="酒店类型")
    estimated_cost: int = Field(default=0, description="预估费用(元/晚)")


class DayPlan(BaseModel):
    """单日行程。"""

    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_index: int = Field(..., description="第几天(从0开始)")
    description: str = Field(default="", description="当日行程描述")
    transportation: str = Field(default="", description="交通方式")
    accommodation: str = Field(default="", description="住宿")
    hotel: Optional[Hotel] = Field(default=None, description="推荐酒店")
    attractions: List[Attraction] = Field(default_factory=list, description="景点列表")
    meals: List[Meal] = Field(default_factory=list, description="餐饮列表")


class WeatherInfo(BaseModel):
    """天气信息。"""

    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_weather: str = Field(default="", description="白天天气")
    night_weather: str = Field(default="", description="夜间天气")
    day_temp: Union[int, str] = Field(default=0, description="白天温度")
    night_temp: Union[int, str] = Field(default=0, description="夜间温度")
    wind_direction: str = Field(default="", description="风向")
    wind_power: str = Field(default="", description="风力")

    @field_validator("day_temp", "night_temp", mode="before")
    @classmethod
    def parse_temperature(cls, v):
        """解析温度：移除 °C/℃/° 等单位，非法时回退 0。"""
        if isinstance(v, str):
            v = v.replace("°C", "").replace("℃", "").replace("°", "").strip()
            try:
                return int(float(v))
            except ValueError:
                return 0
        return v


class Budget(BaseModel):
    """预算信息。"""

    total_attractions: int = Field(default=0, description="景点门票总费用")
    total_hotels: int = Field(default=0, description="酒店总费用")
    total_meals: int = Field(default=0, description="餐饮总费用")
    total_transportation: int = Field(default=0, description="交通总费用")
    total: int = Field(default=0, description="总费用")


class RouteInfo(BaseModel):
    """往返交通段信息（去程 / 返程）。"""
    leg: str = Field(default="", description="段名：去程 / 返程")
    mode: str = Field(default="", description="交通方式：公共交通 / 自驾 等")
    origin: str = Field(default="", description="起点城市")
    destination: str = Field(default="", description="终点城市")
    duration_text: str = Field(default="", description="耗时描述（真实数据，如 约 8 小时）")
    distance_text: str = Field(default="", description="距离描述（真实数据）")
    cost_note: str = Field(default="", description="费用说明（高德不返回票价，写 以实际购票为准）")
    cost: int = Field(default=0, description="该段交通预估费用(元)，仅当数据源返回真实票价时非零（如12306参考票价），严禁估算")

class TripPlan(BaseModel):
    """旅行计划（结构化 JSON 行程）。"""

    city: str = Field(..., description="目的地城市")
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="结束日期")
    days: List[DayPlan] = Field(default_factory=list, description="每日行程")
    weather_info: List[WeatherInfo] = Field(default_factory=list, description="天气信息")
    overall_suggestions: str = Field(default="", description="总体建议")
    budget: Optional[Budget] = Field(default=None, description="预算信息")
    routes: Optional[List[RouteInfo]] = Field(default=None, description="往返交通信息（出发地 → 目的地 → 出发地）")


class TripPlanResponse(BaseModel):
    """旅行计划响应（与旧工程 TripPlanResponse 字段保持 1:1）。"""

    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: Optional[TripPlan] = Field(default=None, description="旅行计划数据")


class TripPlanErrorResponse(BaseModel):
    """错误响应。"""

    success: bool = Field(default=False, description="是否成功")
    message: str = Field(..., description="错误消息")
    error_code: Optional[str] = Field(default=None, description="错误代码")

