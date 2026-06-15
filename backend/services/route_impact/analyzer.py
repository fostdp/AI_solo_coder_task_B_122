"""
Route Impact - 商队路径规划影响分析模块

接入模拟的丝绸之路商队路线 (GIS 数据),
评估不同路径的温湿度暴露对药品总损耗的影响,
给出最佳运输季节和路径建议。

核心逻辑:
  1. 定义丝绸之路关键节点 (城市/驿站) 及其气候模型
  2. 计算各路径段的温湿度暴露指数
  3. 叠加 Arrhenius 模型估算药品损耗率
  4. 综合路径距离 + 药品损耗, 推荐最佳路线与季节
"""
import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Season(Enum):
    SPRING = "春季(3-5月)"
    SUMMER = "夏季(6-8月)"
    AUTUMN = "秋季(9-11月)"
    WINTER = "冬季(12-2月)"


@dataclass
class Waypoint:
    name: str
    lat: float
    lng: float
    elevation: float
    climate_by_season: Dict[str, Dict[str, float]]

    def get_climate(self, season: Season) -> Dict[str, float]:
        return self.climate_by_season.get(season.value, {
            "temperature": 20, "humidity": 50, "light": 300, "aw": 0.45,
        })


@dataclass
class RouteSegment:
    from_waypoint: str
    to_waypoint: str
    distance_km: float
    travel_days: int
    difficulty: float

    @property
    def daily_distance(self) -> float:
        if self.travel_days <= 0:
            return self.distance_km
        return self.distance_km / self.travel_days


@dataclass
class Route:
    name: str
    description: str
    waypoints: List[str]
    segments: List[RouteSegment]

    @property
    def total_distance(self) -> float:
        return sum(s.distance_km for s in self.segments)

    @property
    def total_days(self) -> int:
        return sum(s.travel_days for s in self.segments)


@dataclass
class RouteExposure:
    route_name: str
    season: Season
    avg_temperature: float
    max_temperature: float
    avg_humidity: float
    avg_aw: float
    avg_light: float
    temperature_exposure_index: float
    humidity_exposure_index: float
    total_exposure_index: float


@dataclass
class DrugLossEstimate:
    drug_name: str
    shelf_life_at_origin: float
    shelf_life_on_route: float
    degradation_rate: float
    total_loss_pct: float
    loss_amount: float


@dataclass
class RouteRecommendation:
    route_name: str
    season: Season
    exposure: RouteExposure
    drug_losses: List[DrugLossEstimate]
    total_drug_loss_score: float
    total_distance_km: float
    total_travel_days: int
    recommendation_score: float
    is_recommended: bool
    reason: str


SILKROAD_WAYPOINTS = {
    "长安": Waypoint(
        name="长安", lat=34.26, lng=108.94, elevation=400,
        climate_by_season={
            Season.SPRING.value: {"temperature": 16, "humidity": 55, "light": 450, "aw": 0.45},
            Season.SUMMER.value: {"temperature": 28, "humidity": 70, "light": 700, "aw": 0.58},
            Season.AUTUMN.value: {"temperature": 15, "humidity": 60, "light": 350, "aw": 0.48},
            Season.WINTER.value: {"temperature": -1, "humidity": 45, "light": 150, "aw": 0.38},
        },
    ),
    "兰州": Waypoint(
        name="兰州", lat=36.06, lng=103.83, elevation=1520,
        climate_by_season={
            Season.SPRING.value: {"temperature": 12, "humidity": 40, "light": 500, "aw": 0.40},
            Season.SUMMER.value: {"temperature": 23, "humidity": 55, "light": 680, "aw": 0.48},
            Season.AUTUMN.value: {"temperature": 10, "humidity": 50, "light": 380, "aw": 0.42},
            Season.WINTER.value: {"temperature": -7, "humidity": 35, "light": 180, "aw": 0.35},
        },
    ),
    "敦煌": Waypoint(
        name="敦煌", lat=40.14, lng=94.66, elevation=1139,
        climate_by_season={
            Season.SPRING.value: {"temperature": 14, "humidity": 25, "light": 600, "aw": 0.32},
            Season.SUMMER.value: {"temperature": 27, "humidity": 35, "light": 800, "aw": 0.40},
            Season.AUTUMN.value: {"temperature": 10, "humidity": 30, "light": 450, "aw": 0.35},
            Season.WINTER.value: {"temperature": -9, "humidity": 20, "light": 200, "aw": 0.28},
        },
    ),
    "吐鲁番": Waypoint(
        name="吐鲁番", lat=42.95, lng=89.18, elevation=-154,
        climate_by_season={
            Season.SPRING.value: {"temperature": 20, "humidity": 25, "light": 650, "aw": 0.30},
            Season.SUMMER.value: {"temperature": 38, "humidity": 30, "light": 900, "aw": 0.35},
            Season.AUTUMN.value: {"temperature": 16, "humidity": 35, "light": 500, "aw": 0.33},
            Season.WINTER.value: {"temperature": -8, "humidity": 40, "light": 200, "aw": 0.36},
        },
    ),
    "喀什": Waypoint(
        name="喀什", lat=39.47, lng=75.99, elevation=1289,
        climate_by_season={
            Season.SPRING.value: {"temperature": 15, "humidity": 30, "light": 550, "aw": 0.33},
            Season.SUMMER.value: {"temperature": 28, "humidity": 35, "light": 780, "aw": 0.38},
            Season.AUTUMN.value: {"temperature": 12, "humidity": 35, "light": 420, "aw": 0.35},
            Season.WINTER.value: {"temperature": -6, "humidity": 45, "light": 180, "aw": 0.40},
        },
    ),
    "撒马尔罕": Waypoint(
        name="撒马尔罕", lat=39.65, lng=66.96, elevation=702,
        climate_by_season={
            Season.SPRING.value: {"temperature": 16, "humidity": 50, "light": 500, "aw": 0.42},
            Season.SUMMER.value: {"temperature": 30, "humidity": 30, "light": 800, "aw": 0.35},
            Season.AUTUMN.value: {"temperature": 13, "humidity": 45, "light": 400, "aw": 0.40},
            Season.WINTER.value: {"temperature": -2, "humidity": 60, "light": 150, "aw": 0.50},
        },
    ),
}

SILKROAD_ROUTES = [
    Route(
        name="北道(天山北路)",
        description="长安→兰州→敦煌→吐鲁番→喀什→撒马尔罕",
        waypoints=["长安", "兰州", "敦煌", "吐鲁番", "喀什", "撒马尔罕"],
        segments=[
            RouteSegment("长安", "兰州", 620, 20, 0.5),
            RouteSegment("兰州", "敦煌", 1100, 35, 0.6),
            RouteSegment("敦煌", "吐鲁番", 800, 25, 0.7),
            RouteSegment("吐鲁番", "喀什", 1500, 45, 0.5),
            RouteSegment("喀什", "撒马尔罕", 900, 30, 0.4),
        ],
    ),
    Route(
        name="南道(昆仑山南路)",
        description="长安→兰州→敦煌→(沿昆仑山南麓)→喀什→撒马尔罕",
        waypoints=["长安", "兰州", "敦煌", "喀什", "撒马尔罕"],
        segments=[
            RouteSegment("长安", "兰州", 620, 20, 0.5),
            RouteSegment("兰州", "敦煌", 1100, 35, 0.6),
            RouteSegment("敦煌", "喀什", 1800, 55, 0.8),
            RouteSegment("喀什", "撒马尔罕", 900, 30, 0.4),
        ],
    ),
]


class RouteImpactAnalyzer:
    """路径影响分析器"""

    TEMP_THRESHOLD = 30.0
    HUMIDITY_THRESHOLD = 70.0
    AW_THRESHOLD = 0.55

    def __init__(self, config: Optional[dict] = None):
        self.waypoints = dict(SILKROAD_WAYPOINTS)
        self.routes = list(SILKROAD_ROUTES)

    def compute_exposure(self, route: Route, season: Season) -> RouteExposure:
        temps, hums, aws, lights = [], [], [], []

        for wp_name in route.waypoints:
            wp = self.waypoints.get(wp_name)
            if wp is None:
                continue
            climate = wp.get_climate(season)
            temps.append(climate["temperature"])
            hums.append(climate["humidity"])
            aws.append(climate.get("aw", 0.45))
            lights.append(climate.get("light", 300))

        if not temps:
            return RouteExposure(
                route_name=route.name, season=season,
                avg_temperature=20, max_temperature=20,
                avg_humidity=50, avg_aw=0.45, avg_light=300,
                temperature_exposure_index=0, humidity_exposure_index=0,
                total_exposure_index=0,
            )

        avg_temp = sum(temps) / len(temps)
        max_temp = max(temps)
        avg_hum = sum(hums) / len(hums)
        avg_aw = sum(aws) / len(aws)
        avg_light = sum(lights) / len(lights)

        temp_exposure = 0
        if max_temp > self.TEMP_THRESHOLD:
            temp_exposure += (max_temp - self.TEMP_THRESHOLD) * route.total_days * 0.5
        if avg_temp > 25:
            temp_exposure += (avg_temp - 25) * route.total_days * 0.3

        hum_exposure = 0
        if avg_hum > self.HUMIDITY_THRESHOLD:
            hum_exposure += (avg_hum - self.HUMIDITY_THRESHOLD) * route.total_days * 0.3
        if avg_aw > self.AW_THRESHOLD:
            hum_exposure += (avg_aw - self.AW_THRESHOLD) * 100 * route.total_days * 0.4

        total = temp_exposure + hum_exposure

        return RouteExposure(
            route_name=route.name, season=season,
            avg_temperature=round(avg_temp, 1),
            max_temperature=round(max_temp, 1),
            avg_humidity=round(avg_hum, 1),
            avg_aw=round(avg_aw, 3),
            avg_light=round(avg_light, 1),
            temperature_exposure_index=round(temp_exposure, 2),
            humidity_exposure_index=round(hum_exposure, 2),
            total_exposure_index=round(total, 2),
        )

    def estimate_drug_losses(
        self,
        exposure: RouteExposure,
        drug_params: Dict[str, dict],
        stock_quantities: Optional[Dict[str, float]] = None,
        travel_days: int = 100,
    ) -> List[DrugLossEstimate]:
        R = 8.314
        results = []

        for drug_name, params in drug_params.items():
            arr = params.get("arrhenius", params) if "arrhenius" in params else params
            ea = arr.get("Ea", 75000)
            t_ref = arr.get("T_ref", 298.15)
            sl_ref_months = arr.get("shelf_life_ref_months", 24)

            sl_origin = sl_ref_months * 30.44

            T_route = exposure.avg_temperature + 273.15
            k_ref = -math.log(0.9) / (sl_origin * 86400)
            k_route = k_ref * math.exp(-ea / (R * T_route)) / math.exp(-ea / (R * t_ref))

            if k_route > 0:
                sl_route = -math.log(0.9) / k_route / 86400
            else:
                sl_route = sl_origin

            aw = exposure.avg_aw
            aw_factor = 1.0 if aw <= 0.5 else max(0.1, 1 - 2 * (aw - 0.5) ** 1.5)
            sl_route *= aw_factor

            degradation_rate = k_route if k_route > 0 else 0
            total_loss_pct = min(100, (1 - math.exp(-degradation_rate * travel_days * 86400)) * 100)

            stock = (stock_quantities or {}).get(drug_name, 100)
            loss_amount = stock * total_loss_pct / 100

            results.append(DrugLossEstimate(
                drug_name=drug_name,
                shelf_life_at_origin=round(sl_origin, 1),
                shelf_life_on_route=round(sl_route, 1),
                degradation_rate=round(degradation_rate, 8),
                total_loss_pct=round(total_loss_pct, 2),
                loss_amount=round(loss_amount, 2),
            ))

        return results

    def recommend_routes(
        self,
        drug_params: Dict[str, dict],
        stock_quantities: Optional[Dict[str, float]] = None,
    ) -> List[RouteRecommendation]:
        all_recommendations = []

        for route in self.routes:
            for season in Season:
                exposure = self.compute_exposure(route, season)
                drug_losses = self.estimate_drug_losses(
                    exposure, drug_params, stock_quantities, route.total_days,
                )

                total_loss_score = sum(dl.total_loss_pct for dl in drug_losses) / max(len(drug_losses), 1)
                distance_score = route.total_distance / 5000

                rec_score = 100 - total_loss_score * 0.6 - distance_score * 10 - exposure.total_exposure_index * 0.01
                rec_score = max(0, min(100, rec_score))

                worst_drug = max(drug_losses, key=lambda d: d.total_loss_pct) if drug_losses else None
                reason_parts = []
                if exposure.max_temperature > self.TEMP_THRESHOLD:
                    reason_parts.append(f"最高温{exposure.max_temperature}°C超标")
                if exposure.avg_humidity > self.HUMIDITY_THRESHOLD:
                    reason_parts.append(f"平均湿度{exposure.avg_humidity}%偏高")
                if worst_drug and worst_drug.total_loss_pct > 10:
                    reason_parts.append(f"{worst_drug.drug_name}损耗{worst_drug.total_loss_pct:.1f}%")

                if not reason_parts:
                    reason_parts.append("气候条件适宜药品运输")

                all_recommendations.append(RouteRecommendation(
                    route_name=route.name,
                    season=season,
                    exposure=exposure,
                    drug_losses=drug_losses,
                    total_drug_loss_score=round(total_loss_score, 2),
                    total_distance_km=route.total_distance,
                    total_travel_days=route.total_days,
                    recommendation_score=round(rec_score, 2),
                    is_recommended=False,
                    reason="；".join(reason_parts),
                ))

        all_recommendations.sort(key=lambda r: -r.recommendation_score)
        if all_recommendations:
            all_recommendations[0].is_recommended = True

        return all_recommendations

    def get_gis_heatmap_data(
        self,
        route_name: Optional[str] = None,
        season: Optional[Season] = None,
    ) -> List[dict]:
        season = season or Season.AUTUMN
        heatmap = []

        for name, wp in self.waypoints.items():
            if route_name:
                found = False
                for route in self.routes:
                    if route.name == route_name and name in route.waypoints:
                        found = True
                        break
                if not found:
                    continue

            climate = wp.get_climate(season)
            temp_risk = max(0, (climate["temperature"] - 25) / 20) * 0.5
            hum_risk = max(0, (climate["humidity"] - 50) / 40) * 0.3
            aw_risk = max(0, (climate.get("aw", 0.45) - 0.45) / 0.3) * 0.2
            total_risk = min(1.0, temp_risk + hum_risk + aw_risk)

            heatmap.append({
                "name": name,
                "lat": wp.lat,
                "lng": wp.lng,
                "elevation": wp.elevation,
                "temperature": climate["temperature"],
                "humidity": climate["humidity"],
                "aw": climate.get("aw", 0.45),
                "light": climate.get("light", 300),
                "risk_score": round(total_risk, 4),
                "season": season.value,
            })

        return heatmap


class RouteImpactService:
    """商队路径影响分析服务"""

    def __init__(self, config: Optional[dict] = None):
        self._analyzer = RouteImpactAnalyzer(config)
        self._drug_params: Dict[str, dict] = {}

    async def start(self):
        from shared.config_loader import load_config
        cfg = load_config()
        self._drug_params = cfg.get("drugs", {})
        logger.info("RouteImpactService started (%d routes, %d waypoints)",
                     len(self._analyzer.routes), len(self._analyzer.waypoints))

    async def stop(self):
        logger.info("RouteImpactService stopped")

    def analyze_all_routes(self) -> List[RouteRecommendation]:
        return self._analyzer.recommend_routes(self._drug_params)

    def analyze_route(self, route_name: str, season: Season) -> Optional[RouteRecommendation]:
        route = next((r for r in self._analyzer.routes if r.name == route_name), None)
        if not route:
            return None
        exposure = self._analyzer.compute_exposure(route, season)
        losses = self._analyzer.estimate_drug_losses(exposure, self._drug_params)
        total_loss = sum(dl.total_loss_pct for dl in losses) / max(len(losses), 1)
        return RouteRecommendation(
            route_name=route.name, season=season,
            exposure=exposure, drug_losses=losses,
            total_drug_loss_score=round(total_loss, 2),
            total_distance_km=route.total_distance,
            total_travel_days=route.total_days,
            recommendation_score=round(100 - total_loss * 0.6, 2),
            is_recommended=True,
            reason=f"{season.value}沿{route.name}运输分析",
        )

    def get_gis_heatmap(
        self,
        route_name: Optional[str] = None,
        season: Optional[Season] = None,
    ) -> List[dict]:
        return self._analyzer.get_gis_heatmap_data(route_name, season)
