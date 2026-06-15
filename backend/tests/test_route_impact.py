"""
测试 route_impact 模块 - GIS路径影响分析
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.route_impact.analyzer import (
    RouteImpactAnalyzer, RouteImpactService, Season,
    SILKROAD_WAYPOINTS, SILKROAD_ROUTES,
    RouteExposure, DrugLossEstimate,
)


class TestSilkroadWaypoints:
    def test_six_waypoints(self):
        assert len(SILKROAD_WAYPOINTS) == 6

    def test_waypoint_has_climate_for_all_seasons(self):
        for name, wp in SILKROAD_WAYPOINTS.items():
            for season in Season:
                climate = wp.get_climate(season)
                assert "temperature" in climate
                assert "humidity" in climate

    def test_waypoint_elevation_range(self):
        for name, wp in SILKROAD_WAYPOINTS.items():
            assert -200 <= wp.elevation <= 3000

    def test_turpan_lowest(self):
        assert SILKROAD_WAYPOINTS["吐鲁番"].elevation < 0


class TestSilkroadRoutes:
    def test_at_least_two_routes(self):
        assert len(SILKROAD_ROUTES) >= 2

    def test_route_total_distance(self):
        for route in SILKROAD_ROUTES:
            assert route.total_distance > 0
            assert route.total_days > 0

    def test_northern_route_longer_than_southern(self):
        north = next(r for r in SILKROAD_ROUTES if "北道" in r.name)
        south = next(r for r in SILKROAD_ROUTES if "南道" in r.name)
        assert north.total_distance >= south.total_distance


class TestSeason:
    def test_four_seasons(self):
        assert len(Season) == 4

    def test_season_values(self):
        assert "春季" in Season.SPRING.value
        assert "夏季" in Season.SUMMER.value
        assert "秋季" in Season.AUTUMN.value
        assert "冬季" in Season.WINTER.value


class TestRouteImpactAnalyzer:
    def setup_method(self):
        self.analyzer = RouteImpactAnalyzer()

    def test_compute_exposure_autumn(self):
        route = SILKROAD_ROUTES[0]
        exposure = self.analyzer.compute_exposure(route, Season.AUTUMN)
        assert isinstance(exposure, RouteExposure)
        assert exposure.avg_temperature > -20
        assert exposure.avg_humidity > 0
        assert exposure.total_exposure_index >= 0

    def test_compute_exposure_summer_higher_temp(self):
        route = SILKROAD_ROUTES[0]
        summer = self.analyzer.compute_exposure(route, Season.SUMMER)
        winter = self.analyzer.compute_exposure(route, Season.WINTER)
        assert summer.avg_temperature > winter.avg_temperature

    def test_estimate_drug_losses(self):
        exposure = self.analyzer.compute_exposure(SILKROAD_ROUTES[0], Season.SUMMER)
        drug_params = {
            "当归": {"arrhenius": {"Ea": 85000, "A": 1.2e12, "T_ref": 298.15, "shelf_life_ref_months": 24}},
            "大黄": {"arrhenius": {"Ea": 78000, "A": 8.5e11, "T_ref": 298.15, "shelf_life_ref_months": 18}},
        }
        losses = self.analyzer.estimate_drug_losses(exposure, drug_params, travel_days=100)
        assert len(losses) == 2
        assert all(dl.total_loss_pct >= 0 for dl in losses)

    def test_recommend_routes(self):
        drug_params = {
            "当归": {"arrhenius": {"Ea": 85000, "A": 1.2e12, "T_ref": 298.15, "shelf_life_ref_months": 24}},
        }
        recs = self.analyzer.recommend_routes(drug_params)
        assert len(recs) > 0
        assert recs[0].is_recommended
        for i in range(1, len(recs)):
            assert recs[i].recommendation_score <= recs[0].recommendation_score

    def test_gis_heatmap_data(self):
        data = self.analyzer.get_gis_heatmap_data(season=Season.AUTUMN)
        assert len(data) > 0
        assert all("lat" in d and "lng" in d and "risk_score" in d for d in data)
        assert all(0 <= d["risk_score"] <= 1 for d in data)

    def test_gis_heatmap_filtered_by_route(self):
        data = self.analyzer.get_gis_heatmap_data(route_name="北道(天山北路)", season=Season.AUTUMN)
        assert len(data) > 0
        assert len(data) <= len(SILKROAD_WAYPOINTS)

    def test_turpan_summer_extreme(self):
        turpan = SILKROAD_WAYPOINTS["吐鲁番"]
        summer_climate = turpan.get_climate(Season.SUMMER)
        assert summer_climate["temperature"] > 35


class TestRouteImpactService:
    def test_analyze_all_routes(self):
        svc = RouteImpactService()
        svc._drug_params = {
            "当归": {"arrhenius": {"Ea": 85000, "A": 1.2e12, "T_ref": 298.15, "shelf_life_ref_months": 24}},
        }
        recs = svc.analyze_all_routes()
        assert len(recs) >= 2

    def test_analyze_specific_route(self):
        svc = RouteImpactService()
        svc._drug_params = {
            "当归": {"arrhenius": {"Ea": 85000, "A": 1.2e12, "T_ref": 298.15, "shelf_life_ref_months": 24}},
        }
        rec = svc.analyze_route("北道(天山北路)", Season.AUTUMN)
        assert rec is not None
        assert rec.route_name == "北道(天山北路)"

    def test_analyze_nonexistent_route(self):
        svc = RouteImpactService()
        svc._drug_params = {}
        rec = svc.analyze_route("不存在的路线", Season.AUTUMN)
        assert rec is None

    def test_get_gis_heatmap(self):
        svc = RouteImpactService()
        svc._drug_params = {}
        data = svc.get_gis_heatmap(season=Season.SPRING)
        assert len(data) > 0
