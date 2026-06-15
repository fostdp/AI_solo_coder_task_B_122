"""
[v3] 新增功能模块 API Router
整合: 调配优化 / 方剂替代 / 气候调控 / 路径影响
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from pydantic import BaseModel

from services.allocation_optimizer.optimizer import (
    AllocationService, TentDrugInventory,
)
from services.herb_substitute.knowledge_graph import HerbSubstituteService
from services.climate_control.dqn_controller import ClimateControlService
from services.route_impact.analyzer import RouteImpactService, Season

router = APIRouter(prefix="/api/v3", tags=["v3-advanced"])


# ---- 服务单例 ----

_allocation_svc: Optional[AllocationService] = None
_herb_svc: Optional[HerbSubstituteService] = None
_climate_svc: Optional[ClimateControlService] = None
_route_svc: Optional[RouteImpactService] = None


def init_services(
    allocation_svc: AllocationService,
    herb_svc: HerbSubstituteService,
    climate_svc: ClimateControlService,
    route_svc: RouteImpactService,
):
    global _allocation_svc, _herb_svc, _climate_svc, _route_svc
    _allocation_svc = allocation_svc
    _herb_svc = herb_svc
    _climate_svc = climate_svc
    _route_svc = route_svc


# ==== 请求/响应模型 ====

class AllocationRequest(BaseModel):
    drug_risks: List[dict]
    stock_data: Optional[dict] = None
    tent_distances: Optional[dict] = None


class HerbSubstituteRequest(BaseModel):
    herb_names: List[str]
    max_depth: int = 3
    top_k: int = 5


class ClimateControlRequest(BaseModel):
    tent_id: int
    temperature: float = 25.0
    humidity: float = 60.0
    light: float = 400.0
    aw: float = 0.50


class RouteAnalysisRequest(BaseModel):
    route_name: Optional[str] = None
    season: Optional[str] = None


# ==== 1. 药材调配优化 ====

@router.post("/allocation/optimize")
async def optimize_allocation(req: AllocationRequest):
    if _allocation_svc is None:
        raise HTTPException(status_code=503, detail="Allocation service not initialized")

    inventories = _allocation_svc.build_inventories(
        req.drug_risks, req.stock_data
    )
    distances = None
    if req.tent_distances:
        distances = {}
        for key_str, dist_val in req.tent_distances.items():
            parts = key_str.split(",")
            if len(parts) == 2:
                distances[(int(parts[0]), int(parts[1]))] = float(dist_val)

    result = _allocation_svc.optimize_allocation(inventories, distances)

    return {
        "status": result.status,
        "objective_value": result.objective_value,
        "total_waste_reduction": result.total_waste_reduction,
        "total_transport_cost": result.total_transport_cost,
        "allocations": [
            {
                "drug_name": a.drug_name,
                "from_tent": a.from_tent,
                "to_tent": a.to_tent,
                "quantity": a.quantity,
                "reason": a.reason,
                "estimated_waste_reduction": a.estimated_waste_reduction,
                "estimated_supply_improvement": a.estimated_supply_improvement,
            }
            for a in result.allocations
        ],
        "tent_summaries": result.tent_summaries,
    }


@router.get("/allocation/auto")
async def auto_allocate():
    if _allocation_svc is None:
        raise HTTPException(status_code=503, detail="Allocation service not initialized")

    from shared.config_loader import get_tents, get_drug_params, get_drug_list
    from services.arrhenius_predictor.predictor import ArrheniusPredictor

    tents = get_tents()
    drug_risks = []
    for tent in tents:
        for drug_name in tent.get("drugs", []):
            params = get_drug_params(drug_name)
            if not params:
                continue
            predictor = ArrheniusPredictor(drug_name, params)
            result = predictor.predict_shelf_life(T_celsius=25, aw=0.5, light_lux=0)
            drug_risks.append({
                "tent_id": tent["id"],
                "drug_name": drug_name,
                "shelf_life_days": result["shelf_life_days"],
            })

    inventories = _allocation_svc.build_inventories(drug_risks)
    distances = _allocation_svc.get_tent_distances(tents)
    result = _allocation_svc.optimize_allocation(inventories, distances)

    return {
        "status": result.status,
        "allocations_count": len(result.allocations),
        "total_waste_reduction": result.total_waste_reduction,
        "allocations": [
            {
                "drug_name": a.drug_name,
                "from_tent": a.from_tent,
                "to_tent": a.to_tent,
                "quantity": a.quantity,
                "reason": a.reason,
            }
            for a in result.allocations
        ],
    }


# ==== 2. 方剂替代推荐 ====

@router.post("/herb/substitutes")
async def recommend_substitutes(req: HerbSubstituteRequest):
    if _herb_svc is None:
        raise HTTPException(status_code=503, detail="Herb substitute service not initialized")

    results = {}
    for herb_name in req.herb_names:
        recs = _herb_svc.recommend_substitutes(
            herb_name, max_depth=req.max_depth, top_k=req.top_k
        )
        results[herb_name] = [
            {
                "substitute_herb": r.substitute_herb,
                "similarity_score": r.similarity_score,
                "path_type": r.path_type,
                "path_length": r.path_length,
                "shared_efficacy": r.shared_efficacy,
                "shared_meridians": r.shared_meridians,
                "notes": r.notes,
                "available_in_tents": r.available_in_tents,
            }
            for r in recs
        ]

    return {"original_herbs": req.herb_names, "recommendations": results}


@router.get("/herb/substitutes/{herb_name}")
async def get_single_substitute(
    herb_name: str,
    max_depth: int = Query(default=3, le=5),
    top_k: int = Query(default=5, le=10),
):
    if _herb_svc is None:
        raise HTTPException(status_code=503, detail="Herb substitute service not initialized")

    recs = _herb_svc.recommend_substitutes(herb_name, max_depth=max_depth, top_k=top_k)
    return {
        "original_herb": herb_name,
        "recommendations": [
            {
                "substitute_herb": r.substitute_herb,
                "similarity_score": r.similarity_score,
                "path_type": r.path_type,
                "shared_efficacy": r.shared_efficacy,
                "shared_meridians": r.shared_meridians,
                "notes": r.notes,
                "available_in_tents": r.available_in_tents,
            }
            for r in recs
        ],
    }


@router.get("/herb/graph/neighbors/{herb_name}")
async def get_herb_neighbors(herb_name: str):
    if _herb_svc is None:
        raise HTTPException(status_code=503, detail="Herb substitute service not initialized")

    graph = _herb_svc._graph
    node = graph.get_node(herb_name)
    if not node:
        raise HTTPException(status_code=404, detail=f"Herb '{herb_name}' not found in knowledge graph")

    neighbors = graph.get_neighbors(herb_name)
    return {
        "herb": {
            "name": node.name,
            "nature": node.nature,
            "flavor": node.flavor,
            "meridians": node.meridians,
            "efficacy": node.efficacy,
            "category": node.category,
        },
        "neighbors": [
            {"name": n, "edge_type": et, "weight": w}
            for n, et, w in neighbors
        ],
    }


# ==== 3. 微气候调控策略 ====

@router.post("/climate/recommend")
async def recommend_climate_control(req: ClimateControlRequest):
    if _climate_svc is None:
        raise HTTPException(status_code=503, detail="Climate control service not initialized")

    climate = {
        "temperature": req.temperature,
        "humidity": req.humidity,
        "light": req.light,
        "aw": req.aw,
    }
    rec = _climate_svc.recommend(req.tent_id, climate)

    return {
        "tent_id": rec.tent_id,
        "action": {
            "ventilation": rec.action.ventilation,
            "shading": rec.action.shading,
            "humidifier": rec.action.humidifier,
            "action_index": rec.action.index,
            "description": rec.action.describe(),
            "energy_cost": rec.action.energy_cost(),
        },
        "expected_reward": rec.expected_reward,
        "current_state": {
            "temperature": rec.current_state.temperature,
            "humidity": rec.current_state.humidity,
            "light": rec.current_state.light,
            "aw": rec.current_state.aw,
        },
        "projected_state": {
            "temperature": rec.projected_state.temperature,
            "humidity": rec.projected_state.humidity,
            "light": rec.projected_state.light,
            "aw": rec.projected_state.aw,
        },
        "shelf_life_improvement_days": rec.shelf_life_improvement_days,
        "description": rec.description,
    }


@router.post("/climate/batch-recommend")
async def batch_climate_recommend():
    if _climate_svc is None:
        raise HTTPException(status_code=503, detail="Climate control service not initialized")

    from shared.config_loader import get_tents

    results = []
    for tent in get_tents():
        climate = {"temperature": 25, "humidity": 60, "light": 400, "aw": 0.50}
        rec = _climate_svc.recommend(tent["id"], climate)
        results.append({
            "tent_id": tent["id"],
            "tent_name": tent["name"],
            "action_description": rec.description,
            "shelf_life_improvement_days": rec.shelf_life_improvement_days,
            "expected_reward": rec.expected_reward,
        })

    return {"recommendations": results}


@router.post("/climate/train")
async def train_dqn(episodes: int = Query(default=200, le=1000)):
    if _climate_svc is None:
        raise HTTPException(status_code=503, detail="Climate control service not initialized")

    _climate_svc.pretrain(episodes=episodes)
    return {"status": "ok", "episodes": episodes, "message": f"DQN agent trained with {episodes} episodes"}


# ==== 4. 商队路径影响分析 ====

@router.get("/route/recommendations")
async def get_route_recommendations():
    if _route_svc is None:
        raise HTTPException(status_code=503, detail="Route impact service not initialized")

    recs = _route_svc.analyze_all_routes()
    return {
        "total_recommendations": len(recs),
        "recommendations": [
            {
                "route_name": r.route_name,
                "season": r.season.value,
                "recommendation_score": r.recommendation_score,
                "is_recommended": r.is_recommended,
                "total_distance_km": r.total_distance_km,
                "total_travel_days": r.total_travel_days,
                "total_drug_loss_score": r.total_drug_loss_score,
                "exposure": {
                    "avg_temperature": r.exposure.avg_temperature,
                    "max_temperature": r.exposure.max_temperature,
                    "avg_humidity": r.exposure.avg_humidity,
                    "avg_aw": r.exposure.avg_aw,
                    "total_exposure_index": r.exposure.total_exposure_index,
                },
                "reason": r.reason,
            }
            for r in recs
        ],
    }


@router.get("/route/analyze/{route_name}")
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

    return {
        "route_name": result.route_name,
        "season": result.season.value,
        "recommendation_score": result.recommendation_score,
        "total_distance_km": result.total_distance_km,
        "total_travel_days": result.total_travel_days,
        "exposure": {
            "avg_temperature": result.exposure.avg_temperature,
            "max_temperature": result.exposure.max_temperature,
            "avg_humidity": result.exposure.avg_humidity,
            "avg_aw": result.exposure.avg_aw,
            "temperature_exposure_index": result.exposure.temperature_exposure_index,
            "humidity_exposure_index": result.exposure.humidity_exposure_index,
            "total_exposure_index": result.exposure.total_exposure_index,
        },
        "drug_losses": [
            {
                "drug_name": dl.drug_name,
                "shelf_life_at_origin": dl.shelf_life_at_origin,
                "shelf_life_on_route": dl.shelf_life_on_route,
                "total_loss_pct": dl.total_loss_pct,
                "loss_amount": dl.loss_amount,
            }
            for dl in result.drug_losses
        ],
        "reason": result.reason,
    }


@router.get("/route/gis-heatmap")
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


@router.get("/route/seasons")
async def list_seasons():
    return {"seasons": [s.value for s in Season]}


@router.get("/route/routes")
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
