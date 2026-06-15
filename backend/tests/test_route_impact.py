"""
测试 route_impact 模块 - GIS路径影响分析

覆盖维度:
  - 正常: 沙漠路径药品损耗率>绿洲路径; 冬季损耗低于夏季; 推荐排序正确
  - 边界: 无GIS数据时使用默认温度带; 空路线不崩溃; 单驿站路线
  - 异常: 温湿度数据缺失时线性插值; 驿站不存在时跳过
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.route_impact.analyzer import (
    RouteImpactAnalyzer, RouteImpactService, Season,
    Waypoint, Route, RouteSegment, RouteExposure,
    SILKROAD_WAYPOINTS, SILKROAD_ROUTES,
)


# ============================================================
#  正常场景
# ============================================================

class TestNormalRouteImpact:
    def test_desert_route_loss_higher_than_oasis_route(self):
        analyzer = RouteImpactAnalyzer()
        analyzer.waypoints["沙漠站"] = Waypoint(
            name="沙漠站", lat=42.0, lng=90.0, elevation=500,
            climate_by_season={
                Season.SUMMER.value: {"temperature": 38, "humidity": 20, "light": 900, "aw": 0.25},
            },
        )
        analyzer.waypoints["绿洲站"] = Waypoint(
            name="绿洲站", lat=42.0, lng=90.5, elevation=500,
            climate_by_season={
                Season.SUMMER.value: {"temperature": 28, "humidity": 60, "light": 500, "aw": 0.50},
            },
        )
        desert_route = Route("沙漠路", "test", ["沙漠站"], [RouteSegment("沙漠站", "沙漠站", 100, 5, 0.5)])
        oasis_route = Route("绿洲路", "test", ["绿洲站"], [RouteSegment("绿洲站", "绿洲站", 100, 5, 0.5)])

        desert_exp = analyzer.compute_exposure(desert_route, Season.SUMMER)
        oasis_exp = analyzer.compute_exposure(oasis_route, Season.SUMMER)
        assert desert_exp.avg_temperature > oasis_exp.avg_temperature
        assert desert_exp.temperature_exposure_index > oasis_exp.temperature_exposure_index

    def test_summer_loss_higher_than_winter(self):
        analyzer = RouteImpactAnalyzer()
        drug_params = {
            "当归": {"arrhenius": {"Ea": 85000, "A": 1.2e12, "T_ref": 298.15, "shelf_life_ref_months": 24}},
        }
        summer_exp = analyzer.compute_exposure(SILKROAD_ROUTES[0], Season.SUMMER)
        winter_exp = analyzer.compute_exposure(SILKROAD_ROUTES[0], Season.WINTER)
        summer_losses = analyzer.estimate_drug_losses(summer_exp, drug_params, travel_days=100)
        winter_losses = analyzer.estimate_drug_losses(winter_exp, drug_params, travel_days=100)
        assert summer_losses[0].total_loss_pct >= winter_losses[0].total_loss_pct

    def test_recommendation_sorted_by_score(self):
        analyzer = RouteImpactAnalyzer()
        drug_params = {
            "当归": {"arrhenius": {"Ea": 85000, "A": 1.2e12, "T_ref": 298.15, "shelf_life_ref_months": 24}},
        }
        recs = analyzer.recommend_routes(drug_params)
        assert len(recs) > 0
        assert recs[0].is_recommended
        for i in range(1, len(recs)):
            assert recs[i].recommendation_score <= recs[0].recommendation_score

    def test_autumn_moderate_exposure(self):
        analyzer = RouteImpactAnalyzer()
        exposure = analyzer.compute_exposure(SILKROAD_ROUTES[0], Season.AUTUMN)
        assert 0 < exposure.avg_temperature < 30
        assert exposure.avg_humidity < 80

    def test_turpan_summer_extreme_temp(self):
        turpan = SILKROAD_WAYPOINTS["吐鲁番"]
        climate = turpan.get_climate(Season.SUMMER)
        assert climate["temperature"] > 35

    def test_drug_loss_increases_with_travel_days(self):
        analyzer = RouteImpactAnalyzer()
        drug_params = {
            "当归": {"arrhenius": {"Ea": 85000, "A": 1.2e12, "T_ref": 298.15, "shelf_life_ref_months": 24}},
        }
        exposure = analyzer.compute_exposure(SILKROAD_ROUTES[0], Season.SUMMER)
        short = analyzer.estimate_drug_losses(exposure, drug_params, travel_days=30)
        long = analyzer.estimate_drug_losses(exposure, drug_params, travel_days=200)
        assert long[0].total_loss_pct >= short[0].total_loss_pct


# ============================================================
#  边界场景
# ============================================================

class TestBoundaryRouteImpact:
    def test_no_gis_data_uses_default_temperature_band(self):
        analyzer = RouteImpactAnalyzer()
        analyzer.waypoints = {}
        empty_route = Route("空路", "test", ["A"], [RouteSegment("A", "A", 100, 10, 0.5)])
        for season in Season:
            temps = analyzer._default_temperature_band(empty_route, season)
            assert len(temps) == 1
            assert temps[0] is not None

    def test_empty_route_no_waypoints(self):
        analyzer = RouteImpactAnalyzer()
        empty_route = Route("空路", "test", [], [])
        exposure = analyzer.compute_exposure(empty_route, Season.AUTUMN)
        assert exposure.avg_temperature == 20
        assert exposure.total_exposure_index == 0

    def test_single_waypoint_route(self):
        analyzer = RouteImpactAnalyzer()
        wp = Waypoint("单站", 40.0, 94.0, 1000, {
            Season.AUTUMN.value: {"temperature": 15, "humidity": 40, "light": 400, "aw": 0.35},
        })
        analyzer.waypoints["单站"] = wp
        route = Route("单站路", "test", ["单站"], [RouteSegment("单站", "单站", 50, 5, 0.3)])
        exposure = analyzer.compute_exposure(route, Season.AUTUMN)
        assert exposure.avg_temperature == 15

    def test_default_temperature_band_seasons(self):
        route = Route("test", "t", ["A", "B"], [RouteSegment("A", "B", 100, 10, 0.5)])
        assert RouteImpactAnalyzer._default_temperature_band(route, Season.SPRING)[0] == 15.0
        assert RouteImpactAnalyzer._default_temperature_band(route, Season.SUMMER)[0] == 28.0
        assert RouteImpactAnalyzer._default_temperature_band(route, Season.AUTUMN)[0] == 12.0
        assert RouteImpactAnalyzer._default_temperature_band(route, Season.WINTER)[0] == -5.0

    def test_gis_heatmap_unfiltered_returns_all(self):
        analyzer = RouteImpactAnalyzer()
        data = analyzer.get_gis_heatmap_data(season=Season.AUTUMN)
        assert len(data) == len(SILKROAD_WAYPOINTS)

    def test_gis_heatmap_nonexistent_route_returns_empty(self):
        analyzer = RouteImpactAnalyzer()
        data = analyzer.get_gis_heatmap_data(route_name="不存在的路线", season=Season.AUTUMN)
        assert len(data) == 0


# ============================================================
#  异常场景
# ============================================================

class TestExceptionRouteImpact:
    def test_missing_temperature_uses_interpolation(self):
        RouteImpactAnalyzer._interpolate_missing([None, None, 20.0, None, 30.0, None])
        values = [None, None, 20.0, None, 30.0, None]
        RouteImpactAnalyzer._interpolate_missing(values)
        assert values[0] == 20.0
        assert values[1] == 20.0
        assert values[2] == 20.0
        assert abs(values[3] - 25.0) < 0.01
        assert values[4] == 30.0
        assert values[5] == 30.0

    def test_missing_humidity_uses_interpolation(self):
        values = [50.0, None, None, 70.0]
        RouteImpactAnalyzer._interpolate_missing(values)
        assert abs(values[1] - 56.67) < 0.1
        assert abs(values[2] - 63.33) < 0.1

    def test_all_none_values_unchanged(self):
        values = [None, None, None]
        RouteImpactAnalyzer._interpolate_missing(values)
        assert values == [None, None, None]

    def test_single_value_filled(self):
        values = [None, 25.0, None]
        RouteImpactAnalyzer._interpolate_missing(values)
        assert values[0] == 25.0
        assert values[2] == 25.0

    def test_waypoint_not_in_graph_skipped(self):
        analyzer = RouteImpactAnalyzer()
        route = Route("含未知站", "test", ["长安", "未知站", "兰州"],
                       [RouteSegment("长安", "未知站", 300, 10, 0.5),
                        RouteSegment("未知站", "兰州", 320, 10, 0.5)])
        exposure = analyzer.compute_exposure(route, Season.AUTUMN)
        assert exposure.avg_temperature > -50
        assert exposure.avg_humidity > 0

    def test_analyze_nonexistent_route_returns_none(self):
        svc = RouteImpactService()
        svc._drug_params = {}
        rec = svc.analyze_route("不存在的路线", Season.AUTUMN)
        assert rec is None

    def test_drug_params_with_missing_arrhenius_uses_defaults(self):
        analyzer = RouteImpactAnalyzer()
        exposure = analyzer.compute_exposure(SILKROAD_ROUTES[0], Season.SUMMER)
        drug_params = {"测试药": {"Ea": 70000, "T_ref": 298.15, "shelf_life_ref_months": 24}}
        losses = analyzer.estimate_drug_losses(exposure, drug_params)
        assert len(losses) == 1
        assert losses[0].total_loss_pct >= 0


# ============================================================
#  数据完整性
# ============================================================

class TestSilkroadDataIntegrity:
    def test_six_waypoints(self):
        assert len(SILKROAD_WAYPOINTS) == 6

    def test_waypoint_has_climate_for_all_seasons(self):
        for name, wp in SILKROAD_WAYPOINTS.items():
            for season in Season:
                climate = wp.get_climate(season)
                assert "temperature" in climate
                assert "humidity" in climate

    def test_waypoint_elevation_range(self):
        for wp in SILKROAD_WAYPOINTS.values():
            assert -200 <= wp.elevation <= 3000

    def test_turpan_lowest(self):
        assert SILKROAD_WAYPOINTS["吐鲁番"].elevation < 0

    def test_at_least_two_routes(self):
        assert len(SILKROAD_ROUTES) >= 2

    def test_northern_route_longer(self):
        north = next(r for r in SILKROAD_ROUTES if "北道" in r.name)
        south = next(r for r in SILKROAD_ROUTES if "南道" in r.name)
        assert north.total_distance >= south.total_distance

    def test_four_seasons(self):
        assert len(Season) == 4


# ============================================================
#  服务层
# ============================================================

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

    def test_get_gis_heatmap(self):
        svc = RouteImpactService()
        svc._drug_params = {}
        data = svc.get_gis_heatmap(season=Season.SPRING)
        assert len(data) > 0
