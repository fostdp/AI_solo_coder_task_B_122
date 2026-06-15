"""
路径影响分析子应用 - FastAPI主入口
独立子应用：/api/v3/route
"""
from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List

from services.route_impact.analyzer import RouteImpactService, Season
from .schemas import (
    RouteRecommendation, RouteAnalysisResponse,
    ExposureInfo, DrugLossItem,
)

app = FastAPI(
    title="商队路径影响分析 API",
    description="基于GIS数据的丝绸之路商队路径药品损耗评估服务",
    version="3.0.0",
)

_route_svc: Optional[RouteImpactService] = None


def init_service(service: RouteImpactService):
    global _route_svc
    _route_svc = service


@app.get("/health")
async def health():
    return {"status": "ok", "service": "route_app"}


@app.get("/recommendations")
async def get_route_recommendations():
    if _route_svc is None:
        raise HTTPException(status_code=503, detail="Route impact service not initialized")

    recs = _route_svc.analyze_all_routes()
    return {
        "total_recommendations": len(recs),
        "recommendations": [
            RouteRecommendation(
                route_name=r.route_name,
                season=r.season.value,
                recommendation_score=r.recommendation_score,
                is_recommended=r.is_recommended,
                total_distance_km=r.total_distance_km,
                total_travel_days=r.total_travel_days,
                total_drug_loss_score=r.total_drug_loss_score,
                exposure=ExposureInfo(
                    avg_temperature=r.exposure.avg_temperature,
                    max_temperature=r.exposure.max_temperature,
                    avg_humidity=r.exposure.avg_humidity,
                    avg_aw=r.exposure.avg_aw,
                    total_exposure_index=r.exposure.total_exposure_index,
                ),
                reason=r.reason,
            )
            for r in recs
        ],
    }


@app.get("/analyze/{route_name}", response_model=RouteAnalysisResponse)
async def analyze_route(
    route_name: str,
    season: str = Query(default="秋季(9-11月)"),
):
    if _route_svc is None:
        raise HTTPException(status_code=503, detail="Route impact service not initialized")

    season_enum = None
    for s in Season:
        if s.value == season:
            season_enum = s
            break
    if season_enum is None:
        raise HTTPException(status_code=400, detail=f"Invalid season: {season}")

    result = _route_svc.analyze_route(route_name, season_enum)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Route '{route_name}' not found")

    return RouteAnalysisResponse(
        route_name=result.route_name,
        season=result.season.value,
        recommendation_score=result.recommendation_score,
        total_distance_km=result.total_distance_km,
        total_travel_days=result.total_travel_days,
        exposure=ExposureInfo(
            avg_temperature=result.exposure.avg_temperature,
            max_temperature=result.exposure.max_temperature,
            avg_humidity=result.exposure.avg_humidity,
            avg_aw=result.exposure.avg_aw,
            total_exposure_index=result.exposure.total_exposure_index,
        ),
        drug_losses=[
            DrugLossItem(
                drug_name=dl.drug_name,
                shelf_life_at_origin=dl.shelf_life_at_origin,
                shelf_life_on_route=dl.shelf_life_on_route,
                total_loss_pct=dl.total_loss_pct,
                loss_amount=dl.loss_amount,
            )
            for dl in result.drug_losses
        ],
        reason=result.reason,
    )


@app.get("/gis-heatmap")
async def get_gis_heatmap(
    route_name: Optional[str] = None,
    season: str = Query(default="秋季(9-11月)"),
):
    if _route_svc is None:
        raise HTTPException(status_code=503, detail="Route impact service not initialized")

    season_enum = None
    for s in Season:
        if s.value == season:
            season_enum = s
            break
    if season_enum is None:
        season_enum = Season.AUTUMN

    data = _route_svc.get_gis_heatmap(route_name, season_enum)
    return {"waypoints": data, "season": season_enum.value, "route_filter": route_name}


@app.get("/seasons")
async def list_seasons():
    return {"seasons": [s.value for s in Season]}


@app.get("/routes")
async def list_routes():
    if _route_svc is None:
        raise HTTPException(status_code=503, detail="Route impact service not initialized")

    routes = []
    for route in _route_svc._analyzer.routes:
        routes.append({
            "name": route.name,
            "description": route.description,
            "waypoints": route.waypoints,
            "total_distance_km": route.total_distance,
            "total_travel_days": route.total_days,
            "segments": [
                {
                    "from": s.from_waypoint,
                    "to": s.to_waypoint,
                    "distance_km": s.distance_km,
                    "travel_days": s.travel_days,
                }
                for s in route.segments
            ],
        })
    return {"routes": routes}


@app.get("/interpolation/demo")
async def interpolation_demo():
    """演示线性插值缺失数据的能力"""
    from services.route_impact.analyzer import RouteImpactAnalyzer

    test_values = [None, None, 20.0, None, 25.0, None, 30.0, None, None]
    interpolated = RouteImpactAnalyzer._interpolate_missing(test_values)

    return {
        "original": test_values,
        "interpolated": interpolated,
        "method": "forward_fill + backward_fill + linear_interpolation",
    }
