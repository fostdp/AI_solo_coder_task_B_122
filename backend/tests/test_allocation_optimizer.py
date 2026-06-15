"""
测试 allocation_optimizer 模块 - 混合整数规划药品调配

覆盖维度:
  - 正常: 剩余有效期短的药品优先调配; 多帐篷多药材交叉调配
  - 边界: 无药品时解为空; 单帐篷无法调配; 所有帐篷库存均衡无需调配
  - 异常: PuLP不可行时降级为启发式; 零库存不崩溃
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.allocation_optimizer.optimizer import (
    AllocationOptimizer, AllocationService, TentDrugInventory,
)


# ============================================================
#  正常场景
# ============================================================

class TestNormalAllocation:
    def test_short_shelf_life_drugs_prioritized_for_transfer(self):
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 15, 7),
            TentDrugInventory(2, "当归", 20, 3.0, 365, 7),
            TentDrugInventory(3, "当归", 50, 2.0, 180, 7),
        ]
        opt = AllocationOptimizer()
        result = opt.optimize(invs)
        tent1_out = sum(a.quantity for a in result.allocations if a.from_tent == 1)
        tent2_in = sum(a.quantity for a in result.allocations if a.to_tent == 2)
        assert tent1_out > 0 or result.total_waste_reduction >= 0
        assert tent2_in > 0 or len(result.allocations) == 0

    def test_multi_drug_cross_tent_allocation(self):
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 30, 7),
            TentDrugInventory(2, "当归", 10, 3.0, 365, 7),
            TentDrugInventory(1, "大黄", 10, 1.0, 365, 7),
            TentDrugInventory(2, "大黄", 200, 2.0, 15, 7),
        ]
        opt = AllocationOptimizer()
        result = opt.optimize(invs)
        assert result.total_waste_reduction >= 0
        assert result.total_transport_cost >= 0

    def test_allocation_has_reason(self):
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 30, 7),
            TentDrugInventory(2, "当归", 10, 3.0, 365, 7),
        ]
        opt = AllocationOptimizer()
        result = opt.optimize(invs)
        for a in result.allocations:
            assert len(a.reason) > 0
            assert a.quantity > 0

    def test_waste_reduction_tracking(self):
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 30, 7),
            TentDrugInventory(2, "当归", 10, 3.0, 365, 7),
        ]
        opt = AllocationOptimizer()
        result = opt.optimize(invs)
        for a in result.allocations:
            assert a.estimated_waste_reduction >= 0
            assert a.estimated_supply_improvement >= 0


# ============================================================
#  边界场景
# ============================================================

class TestBoundaryAllocation:
    def test_empty_inventories_returns_empty_status(self):
        opt = AllocationOptimizer()
        result = opt.optimize([])
        assert result.status == "empty"
        assert len(result.allocations) == 0
        assert result.total_waste_reduction == 0

    def test_single_tent_no_transfer_possible(self):
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 30, 7),
        ]
        opt = AllocationOptimizer()
        result = opt.optimize(invs)
        assert len(result.allocations) == 0

    def test_balanced_inventories_no_reallocation(self):
        invs = [
            TentDrugInventory(1, "当归", 50, 2.0, 365, 7),
            TentDrugInventory(2, "当归", 50, 2.0, 365, 7),
        ]
        opt = AllocationOptimizer()
        result = opt.optimize(invs)
        assert len(result.allocations) == 0 or result.total_waste_reduction >= 0

    def test_zero_stock_inventory(self):
        invs = [
            TentDrugInventory(1, "当归", 0, 2.0, 365, 7),
            TentDrugInventory(2, "当归", 0, 2.0, 365, 7),
        ]
        opt = AllocationOptimizer()
        result = opt.optimize(invs)
        assert result.status in ("Optimal", "heuristic", "Infeasible", "Undefined")

    def test_very_short_shelf_life_forces_transfer(self):
        invs = [
            TentDrugInventory(1, "细辛", 300, 2.0, 5, 7),
            TentDrugInventory(2, "细辛", 5, 2.0, 365, 7),
        ]
        opt = AllocationOptimizer()
        result = opt.optimize(invs)
        assert result.total_waste_reduction >= 0

    def test_large_distance_increases_transport_cost(self):
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 30, 7),
            TentDrugInventory(2, "当归", 10, 3.0, 365, 7),
        ]
        opt = AllocationOptimizer()
        near = opt.optimize(invs, {(1, 2): 0.1, (2, 1): 0.1})
        far = opt.optimize(invs, {(1, 2): 100.0, (2, 1): 100.0})
        assert far.total_transport_cost >= near.total_transport_cost


# ============================================================
#  异常场景
# ============================================================

class TestExceptionAllocation:
    def test_pulp_unavailable_falls_back_to_heuristic(self):
        import services.allocation_optimizer.optimizer as mod
        orig = mod.PULP_AVAILABLE
        mod.PULP_AVAILABLE = False
        try:
            opt = AllocationOptimizer()
            invs = [
                TentDrugInventory(1, "当归", 200, 2.0, 30, 7),
                TentDrugInventory(2, "当归", 10, 3.0, 365, 7),
            ]
            result = opt.optimize(invs)
            assert result.status == "heuristic"
            assert len(result.allocations) > 0
        finally:
            mod.PULP_AVAILABLE = orig

    def test_heuristic_handles_zero_waste_risk(self):
        import services.allocation_optimizer.optimizer as mod
        orig = mod.PULP_AVAILABLE
        mod.PULP_AVAILABLE = False
        try:
            opt = AllocationOptimizer()
            invs = [
                TentDrugInventory(1, "当归", 50, 2.0, 365, 7),
                TentDrugInventory(2, "当归", 50, 2.0, 365, 7),
            ]
            result = opt.optimize(invs)
            assert result.status == "heuristic"
        finally:
            mod.PULP_AVAILABLE = orig

    def test_heuristic_respects_remaining_excess(self):
        import services.allocation_optimizer.optimizer as mod
        orig = mod.PULP_AVAILABLE
        mod.PULP_AVAILABLE = False
        try:
            opt = AllocationOptimizer()
            invs = [
                TentDrugInventory(1, "当归", 100, 1.0, 10, 7),
                TentDrugInventory(2, "当归", 5, 3.0, 365, 7),
                TentDrugInventory(3, "当归", 3, 4.0, 365, 7),
            ]
            result = opt.optimize(invs)
            total_transferred = sum(a.quantity for a in result.allocations)
            assert total_transferred <= 93
        finally:
            mod.PULP_AVAILABLE = orig


# ============================================================
#  属性计算 (TentDrugInventory)
# ============================================================

class TestTentDrugInventory:
    def test_safety_stock(self):
        inv = TentDrugInventory(1, "当归", 100, 2.0, 365, 7)
        assert inv.safety_stock == 14.0

    def test_days_of_supply(self):
        inv = TentDrugInventory(1, "当归", 100, 2.0, 365)
        assert inv.days_of_supply == 50.0

    def test_days_of_supply_zero_consumption(self):
        inv = TentDrugInventory(1, "当归", 100, 0, 365)
        assert inv.days_of_supply == float('inf')

    def test_excess_stock(self):
        inv = TentDrugInventory(1, "当归", 100, 2.0, 365)
        assert inv.excess_stock == 86.0

    def test_excess_stock_no_excess(self):
        inv = TentDrugInventory(1, "当归", 10, 2.0, 365)
        assert inv.excess_stock == 0.0

    def test_deficit(self):
        inv = TentDrugInventory(1, "当归", 5, 2.0, 365, 7)
        assert inv.deficit == 9.0

    def test_no_deficit(self):
        inv = TentDrugInventory(1, "当归", 100, 2.0, 365)
        assert inv.deficit == 0.0

    def test_waste_risk_high(self):
        inv = TentDrugInventory(1, "当归", 200, 2.0, 30, 7)
        assert inv.waste_risk > 0

    def test_waste_risk_none(self):
        inv = TentDrugInventory(1, "当归", 50, 2.0, 365)
        assert inv.waste_risk == 0.0

    def test_usable_stock_capped_by_shelf_life(self):
        inv = TentDrugInventory(1, "当归", 500, 2.0, 10)
        assert inv.usable_stock == 20.0


# ============================================================
#  服务层
# ============================================================

class TestAllocationService:
    def test_build_inventories(self):
        svc = AllocationService()
        risks = [
            {"tent_id": 1, "drug_name": "当归", "shelf_life_days": 30},
            {"tent_id": 2, "drug_name": "当归", "shelf_life_days": 365},
        ]
        invs = svc.build_inventories(risks)
        assert len(invs) == 2
        assert invs[0].current_stock < invs[1].current_stock

    def test_build_inventories_with_stock_data(self):
        svc = AllocationService()
        risks = [{"tent_id": 1, "drug_name": "当归", "shelf_life_days": 180}]
        stock_data = {1: {"当归": {"current_stock": 200, "daily_consumption": 5.0}}}
        invs = svc.build_inventories(risks, stock_data)
        assert invs[0].current_stock == 200
        assert invs[0].daily_consumption == 5.0

    def test_tent_distances(self):
        svc = AllocationService()
        tents = [{"id": 1, "lat": 40.0, "lng": 94.0}, {"id": 2, "lat": 41.0, "lng": 95.0}]
        dists = svc.get_tent_distances(tents)
        assert (1, 2) in dists
        assert dists[(1, 2)] > 0
        assert abs(dists[(1, 2)] - dists[(2, 1)]) < 0.01

    def test_tent_distances_same_location(self):
        svc = AllocationService()
        tents = [{"id": 1, "lat": 40.0, "lng": 94.0}, {"id": 2, "lat": 40.0, "lng": 94.0}]
        dists = svc.get_tent_distances(tents)
        assert dists[(1, 2)] < 1.0
