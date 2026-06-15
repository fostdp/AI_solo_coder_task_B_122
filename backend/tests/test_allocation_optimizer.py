"""
测试 allocation_optimizer 模块 - 混合整数规划药品调配
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.allocation_optimizer.optimizer import (
    AllocationOptimizer, AllocationService, TentDrugInventory, AllocationPlan,
)


class TestTentDrugInventory:
    def test_safety_stock(self):
        inv = TentDrugInventory(
            tent_id=1, drug_name="当归", current_stock=100,
            daily_consumption=2.0, shelf_life_days=365, safety_stock_days=7,
        )
        assert inv.safety_stock == 14.0

    def test_days_of_supply(self):
        inv = TentDrugInventory(
            tent_id=1, drug_name="当归", current_stock=100,
            daily_consumption=2.0, shelf_life_days=365,
        )
        assert inv.days_of_supply == 50.0

    def test_days_of_supply_zero_consumption(self):
        inv = TentDrugInventory(
            tent_id=1, drug_name="当归", current_stock=100,
            daily_consumption=0, shelf_life_days=365,
        )
        assert inv.days_of_supply == float('inf')

    def test_excess_stock(self):
        inv = TentDrugInventory(
            tent_id=1, drug_name="当归", current_stock=100,
            daily_consumption=2.0, shelf_life_days=365,
        )
        assert inv.excess_stock == 86.0

    def test_excess_stock_no_excess(self):
        inv = TentDrugInventory(
            tent_id=1, drug_name="当归", current_stock=10,
            daily_consumption=2.0, shelf_life_days=365,
        )
        assert inv.excess_stock == 0.0

    def test_deficit(self):
        inv = TentDrugInventory(
            tent_id=1, drug_name="当归", current_stock=5,
            daily_consumption=2.0, shelf_life_days=365, safety_stock_days=7,
        )
        assert inv.deficit == 9.0

    def test_no_deficit(self):
        inv = TentDrugInventory(
            tent_id=1, drug_name="当归", current_stock=100,
            daily_consumption=2.0, shelf_life_days=365,
        )
        assert inv.deficit == 0.0

    def test_waste_risk_high(self):
        inv = TentDrugInventory(
            tent_id=1, drug_name="当归", current_stock=200,
            daily_consumption=2.0, shelf_life_days=30, safety_stock_days=7,
        )
        assert inv.waste_risk > 0

    def test_waste_risk_none(self):
        inv = TentDrugInventory(
            tent_id=1, drug_name="当归", current_stock=50,
            daily_consumption=2.0, shelf_life_days=365,
        )
        assert inv.waste_risk == 0.0

    def test_usable_stock_capped_by_shelf_life(self):
        inv = TentDrugInventory(
            tent_id=1, drug_name="当归", current_stock=500,
            daily_consumption=2.0, shelf_life_days=10,
        )
        assert inv.usable_stock == 20.0


class TestAllocationOptimizer:
    def _make_inventories(self):
        return [
            TentDrugInventory(1, "当归", 200, 2.0, 30, 7),
            TentDrugInventory(2, "当归", 20, 3.0, 365, 7),
            TentDrugInventory(3, "当归", 50, 2.0, 180, 7),
        ]

    def test_basic_optimization(self):
        opt = AllocationOptimizer()
        result = opt.optimize(self._make_inventories())
        assert result.status in ("Optimal", "heuristic", "Infeasible", "Undefined")

    def test_empty_inventories(self):
        opt = AllocationOptimizer()
        result = opt.optimize([])
        assert result.status == "empty"

    def test_heuristic_fallback(self):
        import services.allocation_optimizer.optimizer as mod
        orig = mod.PULP_AVAILABLE
        mod.PULP_AVAILABLE = False
        try:
            opt = AllocationOptimizer()
            result = opt.optimize(self._make_inventories())
            assert result.status == "heuristic"
        finally:
            mod.PULP_AVAILABLE = orig

    def test_with_distances(self):
        opt = AllocationOptimizer()
        distances = {(1, 2): 0.5, (2, 1): 0.5, (1, 3): 1.0, (3, 1): 1.0}
        result = opt.optimize(self._make_inventories(), tent_distances=distances)
        assert result.total_transport_cost >= 0

    def test_no_reallocation_needed(self):
        invs = [
            TentDrugInventory(1, "当归", 50, 2.0, 365, 7),
            TentDrugInventory(2, "当归", 50, 2.0, 365, 7),
        ]
        opt = AllocationOptimizer()
        result = opt.optimize(invs)
        assert len(result.allocations) == 0 or result.total_waste_reduction >= 0


class TestAllocationService:
    def test_build_inventories(self):
        svc = AllocationService()
        risks = [
            {"tent_id": 1, "drug_name": "当归", "shelf_life_days": 30},
            {"tent_id": 2, "drug_name": "当归", "shelf_life_days": 365},
        ]
        invs = svc.build_inventories(risks)
        assert len(invs) == 2
        assert invs[0].shelf_life_days == 30
        assert invs[0].current_stock < invs[1].current_stock

    def test_build_inventories_with_stock_data(self):
        svc = AllocationService()
        risks = [
            {"tent_id": 1, "drug_name": "当归", "shelf_life_days": 180},
        ]
        stock_data = {1: {"当归": {"current_stock": 200, "daily_consumption": 5.0}}}
        invs = svc.build_inventories(risks, stock_data)
        assert invs[0].current_stock == 200
        assert invs[0].daily_consumption == 5.0

    def test_tent_distances(self):
        svc = AllocationService()
        tents = [
            {"id": 1, "lat": 40.0, "lng": 94.0},
            {"id": 2, "lat": 41.0, "lng": 95.0},
        ]
        dists = svc.get_tent_distances(tents)
        assert (1, 2) in dists
        assert (2, 1) in dists
        assert dists[(1, 2)] > 0
        assert abs(dists[(1, 2)] - dists[(2, 1)]) < 0.01
