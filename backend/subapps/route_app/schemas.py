"""
路径影响分析子应用 - 请求/响应模型
"""
from pydantic import BaseModel
from typing import Optional, List


class RouteAnalysisRequest(BaseModel):
    route_name: Optional[str] = None
    season: Optional[str] = None


class ExposureInfo(BaseModel):
    avg_temperature: float
    max_temperature: float
    avg_humidity: float
    avg_aw: float
    total_exposure_index: float


class DrugLossItem(BaseModel):
    drug_name: str
    shelf_life_at_origin: float
    shelf_life_on_route: float
    total_loss_pct: float
    loss_amount: float


class RouteRecommendation(BaseModel):
    route_name: str
    season: str
    recommendation_score: float
    is_recommended: bool
    total_distance_km: float
    total_travel_days: float
    total_drug_loss_score: float
    exposure: ExposureInfo
    reason: str


class RouteAnalysisResponse(BaseModel):
    route_name: str
    season: str
    recommendation_score: float
    total_distance_km: float
    total_travel_days: float
    exposure: ExposureInfo
    drug_losses: List[DrugLossItem]
    reason: str
