"""
调配优化子应用 - FastAPI主入口
独立子应用：/api/v3/allocation
"""
import time
from fastapi import FastAPI, HTTPException, Query
from typing import Optional

from services.allocation_optimizer.optimizer import (
    AllocationService, AllocationOptimizer,
)
from .schemas import AllocationRequest, AllocationResponse, AllocationItem

app = FastAPI(
    title="药材调配优化 API",
    description="基于混合整数规划的丝绸之路医疗帐篷药品调配优化服务",
    version="3.0.0",
)

_allocation_svc: Optional[AllocationService] = None
_optimizer: Optional[AllocationOptimizer] = None


def init_service(service: AllocationService, optimizer: AllocationOptimizer = None):
    global _allocation_svc, _optimizer
    _allocation_svc = service
    _optimizer = optimizer or AllocationOptimizer()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "allocation_app"}


@app.post("/optimize", response_model=AllocationResponse)
async def optimize_allocation(req: AllocationRequest):
    if _allocation_svc is None:
        raise HTTPException(status_code=503, detail="Allocation service not initialized")

    start = time.time()

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
    solve_ms = (time.time() - start) * 1000

    return AllocationResponse(
        status=result.status,
        objective_value=result.objective_value,
        total_waste_reduction=result.total_waste_reduction,
        total_transport_cost=result.total_transport_cost,
        allocations=[
            AllocationItem(
                drug_name=a.drug_name,
                from_tent=a.from_tent,
                to_tent=a.to_tent,
                quantity=a.quantity,
                reason=a.reason,
                estimated_waste_reduction=a.estimated_waste_reduction,
                estimated_supply_improvement=a.estimated_supply_improvement,
            )
            for a in result.allocations
        ],
        tent_summaries=result.tent_summaries,
        method="column_generation" if result.status.startswith("CG-") else "full_mip",
        solve_time_ms=round(solve_ms, 2),
    )


@app.get("/auto")
async def auto_allocate():
    if _allocation_svc is None:
        raise HTTPException(status_code=503, detail="Allocation service not initialized")

    from shared.config_loader import get_tents, get_drug_params
    from services.arrhenius_predictor.predictor import ArrheniusPredictor

    start = time.time()

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
    solve_ms = (time.time() - start) * 1000

    return {
        "status": result.status,
        "allocations_count": len(result.allocations),
        "total_waste_reduction": result.total_waste_reduction,
        "solve_time_ms": round(solve_ms, 2),
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


@app.get("/method")
async def get_method_info():
    if _optimizer is None:
        return {"use_column_generation": True, "description": "列生成法默认启用"}
    return {
        "use_column_generation": _optimizer.use_column_generation,
        "max_columns_per_drug": _optimizer.MAX_COLUMNS_PER_DRUG,
        "cg_max_iter": _optimizer.COLUMN_GENERATION_MAX_ITER,
    }


@app.get("/stats")
async def get_optimizer_stats():
    return {
        "transport_cost_per_unit": AllocationOptimizer.TRANSPORT_COST_PER_UNIT,
        "waste_penalty_per_unit": AllocationOptimizer.WASTE_PENALTY_PER_UNIT,
        "deficit_penalty_per_unit": AllocationOptimizer.DEFICIT_PENALTY_PER_UNIT,
    }
