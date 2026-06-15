"""
调配优化子应用 - 请求/响应模型
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class AllocationRequest(BaseModel):
    drug_risks: List[dict]
    stock_data: Optional[dict] = None
    tent_distances: Optional[dict] = None


class AllocationItem(BaseModel):
    drug_name: str
    from_tent: int
    to_tent: int
    quantity: float
    reason: str
    estimated_waste_reduction: float
    estimated_supply_improvement: float


class AllocationResponse(BaseModel):
    status: str
    objective_value: float
    total_waste_reduction: float
    total_transport_cost: float
    allocations: List[AllocationItem]
    tent_summaries: Optional[Dict[str, Any]] = None
    method: Optional[str] = None
    solve_time_ms: Optional[float] = None
