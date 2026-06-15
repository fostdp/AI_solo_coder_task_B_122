"""
Allocation Optimizer - 混合整数规划药品调配优化器

基于 PuLP 求解器, 以最小化过期浪费为目标,
生成各帐篷间的药品调配方案。

决策变量: x[i][j][d] = 从帐篷i调拨到帐篷j的药材d的数量 (整数)
目标函数: min Σ (过期浪费量 + 调拨运输成本)
约束条件:
  - 每个帐篷调出量 ≤ 库存 - 安全存量
  - 每个帐篷调入量 ≤ 需求缺口
  - 供需平衡
  - 调拨量非负整数
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from pulp import (
        LpMinimize, LpProblem, LpVariable, lpSum, LpInteger,
        LpStatus, value, PULP_CBC_CMD,
    )
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TentDrugInventory:
    tent_id: int
    drug_name: str
    current_stock: float
    daily_consumption: float
    shelf_life_days: float
    safety_stock_days: float = 7.0

    @property
    def safety_stock(self) -> float:
        return self.daily_consumption * self.safety_stock_days

    @property
    def days_of_supply(self) -> float:
        if self.daily_consumption <= 0:
            return float('inf')
        return self.current_stock / self.daily_consumption

    @property
    def usable_stock(self) -> float:
        max_usable_days = min(self.shelf_life_days, 90)
        return min(self.current_stock, self.daily_consumption * max_usable_days)

    @property
    def excess_stock(self) -> float:
        surplus = self.current_stock - self.safety_stock
        return max(0, surplus)

    @property
    def deficit(self) -> float:
        if self.days_of_supply <= self.safety_stock_days:
            return self.safety_stock - self.current_stock
        return 0.0

    @property
    def waste_risk(self) -> float:
        if self.shelf_life_days <= 0:
            return 1.0
        if self.days_of_supply > self.shelf_life_days:
            expired_frac = (self.days_of_supply - self.shelf_life_days) / self.days_of_supply
            return min(1.0, expired_frac)
        return 0.0


@dataclass
class AllocationPlan:
    drug_name: str
    from_tent: int
    to_tent: int
    quantity: float
    reason: str

    estimated_waste_reduction: float = 0.0
    estimated_supply_improvement: float = 0.0


@dataclass
class OptimizationResult:
    status: str
    total_waste_reduction: float
    total_transport_cost: float
    objective_value: float
    allocations: List[AllocationPlan] = field(default_factory=list)
    tent_summaries: Dict[int, Dict] = field(default_factory=dict)


class AllocationOptimizer:
    """
    混合整数规划药品调配优化器

    目标: 最小化 总过期浪费量 + 调拨运输成本
    约束: 供需平衡 + 安全库存 + 整数调拨量
    """

    TRANSPORT_COST_PER_UNIT = 0.5
    WASTE_PENALTY_PER_UNIT = 10.0
    DEFICIT_PENALTY_PER_UNIT = 8.0

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.transport_cost = cfg.get("transport_cost_per_unit", self.TRANSPORT_COST_PER_UNIT)
        self.waste_penalty = cfg.get("waste_penalty_per_unit", self.WASTE_PENALTY_PER_UNIT)
        self.deficit_penalty = cfg.get("deficit_penalty_per_unit", self.DEFICIT_PENALTY_PER_UNIT)

    def optimize(
        self,
        inventories: List[TentDrugInventory],
        tent_distances: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> OptimizationResult:
        if not inventories:
            return OptimizationResult(
                status="empty", total_waste_reduction=0,
                total_transport_cost=0, objective_value=0,
            )

        if not PULP_AVAILABLE:
            logger.warning("PuLP not available, falling back to heuristic allocation")
            return self._heuristic_optimize(inventories, tent_distances)

        drugs = list({inv.drug_name for inv in inventories})
        tents = list({inv.tent_id for inv in inventories})

        drug_inventories: Dict[str, Dict[int, TentDrugInventory]] = {}
        for inv in inventories:
            drug_inventories.setdefault(inv.drug_name, {})[inv.tent_id] = inv

        prob = LpProblem("DrugAllocation", LpMinimize)

        x = {}
        for d in drugs:
            for i in tents:
                for j in tents:
                    if i != j:
                        var_name = f"x_{i}_{j}_{d}"
                        x[(i, j, d)] = LpVariable(var_name, lowBound=0, cat=LpInteger)

        waste_vars = {}
        deficit_vars = {}
        for d in drugs:
            for t in tents:
                inv = drug_inventories.get(d, {}).get(t)
                if inv is None:
                    continue
                waste_vars[(t, d)] = LpVariable(
                    f"waste_{t}_{d}", lowBound=0, cat=LpInteger
                )
                deficit_vars[(t, d)] = LpVariable(
                    f"deficit_{t}_{d}", lowBound=0, cat=LpInteger
                )

        transport_cost_expr = []
        for (i, j, d), var in x.items():
            dist = 1.0
            if tent_distances and (i, j) in tent_distances:
                dist = tent_distances[(i, j)]
            transport_cost_expr.append(self.transport_cost * dist * var)

        waste_cost_expr = [self.waste_penalty * w for w in waste_vars.values()]
        deficit_cost_expr = [self.deficit_penalty * df for df in deficit_vars.values()]

        prob += lpSum(transport_cost_expr + waste_cost_expr + deficit_cost_expr)

        for d in drugs:
            for t in tents:
                inv = drug_inventories.get(d, {}).get(t)
                if inv is None:
                    continue

                outflow = lpSum(x.get((t, j, d), 0) for j in tents if j != t)
                inflow = lpSum(x.get((i, t, d), 0) for i in tents if i != t)

                net_stock = inv.current_stock - outflow + inflow

                prob += waste_vars[(t, d)] >= net_stock - inv.usable_stock, \
                    f"waste_def_{t}_{d}"

                prob += deficit_vars[(t, d)] >= inv.safety_stock - net_stock, \
                    f"deficit_def_{t}_{d}"

                prob += outflow <= inv.excess_stock, \
                    f"supply_limit_{t}_{d}"

        solver = PULP_CBC_CMD(msg=0, timeLimit=30)
        prob.solve(solver)

        status = LpStatus.get(prob.status, "Unknown")
        obj_val = value(prob.objective) if prob.objective else 0

        allocations = []
        total_waste_red = 0.0
        total_transport = 0.0

        for (i, j, d), var in x.items():
            qty = value(var) if var is not None else 0
            if qty and qty > 0:
                dist = 1.0
                if tent_distances and (i, j) in tent_distances:
                    dist = tent_distances[(i, j)]
                inv_i = drug_inventories.get(d, {}).get(i)
                inv_j = drug_inventories.get(d, {}).get(j)

                reason_parts = []
                if inv_i and inv_i.waste_risk > 0.1:
                    reason_parts.append(
                        f"源帐篷过期风险{inv_i.waste_risk:.0%}"
                    )
                if inv_j and inv_j.deficit > 0:
                    reason_parts.append(
                        f"目标帐篷缺货{inv_j.deficit:.1f}单位"
                    )

                alloc = AllocationPlan(
                    drug_name=d,
                    from_tent=i,
                    to_tent=j,
                    quantity=qty,
                    reason="；".join(reason_parts) if reason_parts else "调配优化",
                    estimated_waste_reduction=qty * min(1, inv_i.waste_risk if inv_i else 0),
                    estimated_supply_improvement=min(qty, inv_j.deficit if inv_j else 0),
                )
                allocations.append(alloc)
                total_waste_red += alloc.estimated_waste_reduction
                total_transport += self.transport_cost * dist * qty

        tent_summaries = {}
        for t in tents:
            summary = {"tent_id": t, "allocations_out": 0, "allocations_in": 0}
            for a in allocations:
                if a.from_tent == t:
                    summary["allocations_out"] += a.quantity
                if a.to_tent == t:
                    summary["allocations_in"] += a.quantity
            tent_summaries[t] = summary

        return OptimizationResult(
            status=status,
            total_waste_reduction=total_waste_red,
            total_transport_cost=total_transport,
            objective_value=obj_val or 0,
            allocations=allocations,
            tent_summaries=tent_summaries,
        )

    def _heuristic_optimize(
        self,
        inventories: List[TentDrugInventory],
        tent_distances: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> OptimizationResult:
        drugs = list({inv.drug_name for inv in inventories})
        drug_inventories: Dict[str, Dict[int, TentDrugInventory]] = {}
        for inv in inventories:
            drug_inventories.setdefault(inv.drug_name, {})[inv.tent_id] = inv

        allocations = []
        total_waste_red = 0.0
        total_transport = 0.0

        remaining_excess: Dict[Tuple[int, str], float] = {}
        for inv in inventories:
            if inv.excess_stock > 0:
                remaining_excess[(inv.tent_id, inv.drug_name)] = inv.excess_stock

        for d in drugs:
            suppliers = []
            demanders = []
            for t, inv in drug_inventories.get(d, {}).items():
                avail = remaining_excess.get((t, d), 0)
                if avail > 0 and inv.waste_risk > 0.1:
                    suppliers.append(inv)
                if inv.deficit > 0:
                    demanders.append(inv)

            for dem in sorted(demanders, key=lambda x: x.deficit, reverse=True):
                remaining_need = dem.deficit
                for sup in sorted(suppliers, key=lambda x: x.waste_risk, reverse=True):
                    if remaining_need <= 0:
                        break
                    sup_avail = remaining_excess.get((sup.tent_id, d), 0)
                    transfer_qty = min(remaining_need, sup_avail)
                    if transfer_qty > 0:
                        dist = 1.0
                        if tent_distances:
                            dist = tent_distances.get((sup.tent_id, dem.tent_id), 1.0)
                        alloc = AllocationPlan(
                            drug_name=d,
                            from_tent=sup.tent_id,
                            to_tent=dem.tent_id,
                            quantity=transfer_qty,
                            reason=f"启发式调配: 过剩{sup_avail:.0f}→缺货{dem.deficit:.0f}",
                            estimated_waste_reduction=transfer_qty * sup.waste_risk,
                            estimated_supply_improvement=transfer_qty,
                        )
                        allocations.append(alloc)
                        total_waste_red += alloc.estimated_waste_reduction
                        total_transport += self.transport_cost * dist * transfer_qty
                        remaining_excess[(sup.tent_id, d)] = sup_avail - transfer_qty
                        remaining_need -= transfer_qty

        return OptimizationResult(
            status="heuristic",
            total_waste_reduction=total_waste_red,
            total_transport_cost=total_transport,
            objective_value=total_waste_red * self.waste_penalty - total_transport,
            allocations=allocations,
        )


class AllocationService:
    """调配优化服务 - 整合库存数据与优化求解"""

    def __init__(self, config: Optional[dict] = None):
        self._optimizer = AllocationOptimizer(config)
        self._ch_client = None
        self._db_name = None

    async def start(self):
        from shared.config_loader import get_clickhouse_config
        from shared.clickhouse_client import get_client
        ch_cfg = get_clickhouse_config()
        self._ch_client = get_client(
            host=ch_cfg["host"], port=ch_cfg["port"],
            user=ch_cfg["user"], password=ch_cfg["password"],
            database=ch_cfg["database"],
        )
        self._db_name = ch_cfg["database"]
        logger.info("AllocationService started")

    async def stop(self):
        logger.info("AllocationService stopped")

    def build_inventories(
        self,
        drug_risks: List[dict],
        stock_data: Optional[Dict[int, Dict[str, dict]]] = None,
    ) -> List[TentDrugInventory]:
        inventories = []
        for risk in drug_risks:
            tent_id = risk["tent_id"]
            drug_name = risk["drug_name"]
            shelf_life = risk.get("shelf_life_days", 365)

            if stock_data and tent_id in stock_data and drug_name in stock_data[tent_id]:
                sd = stock_data[tent_id][drug_name]
                current_stock = sd.get("current_stock", 100)
                daily_consumption = sd.get("daily_consumption", 2.0)
                safety_stock_days = sd.get("safety_stock_days", 7.0)
            else:
                urgency = max(0, 1 - shelf_life / 365)
                current_stock = 100 * (1 - urgency * 0.5)
                daily_consumption = 2.0
                safety_stock_days = 7.0

            inventories.append(TentDrugInventory(
                tent_id=tent_id,
                drug_name=drug_name,
                current_stock=current_stock,
                daily_consumption=daily_consumption,
                shelf_life_days=shelf_life,
                safety_stock_days=safety_stock_days,
            ))
        return inventories

    def optimize_allocation(
        self,
        inventories: List[TentDrugInventory],
        tent_distances: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> OptimizationResult:
        return self._optimizer.optimize(inventories, tent_distances)

    def get_tent_distances(self, tents: List[dict]) -> Dict[Tuple[int, int], float]:
        import math
        distances = {}
        for i, t1 in enumerate(tents):
            for j, t2 in enumerate(tents):
                if i < j:
                    lat1, lng1 = t1["lat"], t1["lng"]
                    lat2, lng2 = t2["lat"], t2["lng"]
                    dlat = math.radians(lat2 - lat1)
                    dlng = math.radians(lng2 - lng1)
                    a = (math.sin(dlat / 2) ** 2 +
                         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                         math.sin(dlng / 2) ** 2)
                    c = 2 * math.asin(math.sqrt(a))
                    km = 6371 * c
                    distances[(t1["id"], t2["id"])] = km
                    distances[(t2["id"], t1["id"])] = km
        return distances
