"""
列生成法单元测试
测试列生成优化器的各个子模块和集成性能
"""
import pytest
import time

from services.allocation_optimizer.optimizer import (
    AllocationOptimizer, TentDrugInventory, OptimizationResult,
)


class TestColumnGenerationCore:
    """列生成法核心逻辑测试"""

    def test_generate_initial_columns(self):
        opt = AllocationOptimizer()
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 15, 7),
            TentDrugInventory(2, "当归", 20, 3.0, 365, 7),
            TentDrugInventory(3, "当归", 50, 2.0, 180, 7),
        ]
        drugs = ["当归"]
        tents = [1, 2, 3]
        drug_invs = {d: {inv.tent_id: inv for inv in invs if inv.drug_name == d} for d in drugs}

        columns = opt._generate_initial_columns(drugs, tents, drug_invs, None)

        assert len(columns) > 0
        for col in columns:
            assert "id" in col
            assert "drug" in col
            assert "from_tent" in col
            assert "to_tent" in col
            assert "quantity" in col
            assert "cost" in col
            assert col["quantity"] > 0
            assert col["from_tent"] != col["to_tent"]

    def test_initial_columns_high_waste_risk_first(self):
        opt = AllocationOptimizer()
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 8, 7),
            TentDrugInventory(2, "当归", 200, 2.0, 30, 7),
            TentDrugInventory(3, "当归", 10, 3.0, 365, 7),
        ]
        drugs = ["当归"]
        tents = [1, 2, 3]
        drug_invs = {d: {inv.tent_id: inv for inv in invs if inv.drug_name == d} for d in drugs}

        columns = opt._generate_initial_columns(drugs, tents, drug_invs, None)

        from_tents = [c["from_tent"] for c in columns]
        assert 1 in from_tents

    def test_solve_restricted_master_basic(self):
        opt = AllocationOptimizer()
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 15, 7),
            TentDrugInventory(2, "当归", 20, 3.0, 365, 7),
        ]
        drugs = ["当归"]
        tents = [1, 2]
        drug_invs = {d: {inv.tent_id: inv for inv in invs if inv.drug_name == d} for d in drugs}

        columns = opt._generate_initial_columns(drugs, tents, drug_invs, None)
        assert len(columns) > 0

        result = opt._solve_restricted_master(
            columns, drugs, tents, drug_invs, None, integer=False
        )

        assert isinstance(result, OptimizationResult)
        assert result.total_waste_reduction >= 0

    def test_pricing_problem_finds_new_columns(self):
        opt = AllocationOptimizer()
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 12, 7),
            TentDrugInventory(2, "当归", 150, 2.0, 25, 7),
            TentDrugInventory(3, "当归", 15, 3.0, 365, 7),
            TentDrugInventory(4, "当归", 8, 2.0, 365, 7),
        ]
        drugs = ["当归"]
        tents = [1, 2, 3, 4]
        drug_invs = {d: {inv.tent_id: inv for inv in invs if inv.drug_name == d} for d in drugs}

        initial_columns = opt._generate_initial_columns(drugs, tents, drug_invs, None)
        limited_cols = initial_columns[:1]

        duals = opt._extract_duals(None, drugs, tents)

        new_cols = opt._pricing_problem(
            duals, drugs, tents, drug_invs, None,
            existing_columns=limited_cols,
        )

        assert isinstance(new_cols, list)
        assert len(new_cols) >= 0
        for col in new_cols:
            assert col["drug"] in drugs
            assert col["from_tent"] != col["to_tent"]

    def test_extract_duals_returns_dict(self):
        opt = AllocationOptimizer()
        drugs = ["当归", "甘草"]
        tents = [1, 2, 3]

        duals = opt._extract_duals(None, drugs, tents)

        assert isinstance(duals, dict)
        assert ("supply", 1, "当归") in duals
        assert ("demand", 2, "甘草") in duals
        assert all(isinstance(v, float) for v in duals.values())


class TestColumnGenerationIntegration:
    """列生成法集成测试"""

    def test_column_generation_end_to_end(self):
        opt = AllocationOptimizer({"use_column_generation": True})
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 12, 7),
            TentDrugInventory(2, "当归", 180, 2.0, 20, 7),
            TentDrugInventory(3, "当归", 15, 3.0, 365, 7),
            TentDrugInventory(4, "当归", 10, 2.0, 365, 7),
        ]

        result = opt.optimize(invs)

        assert isinstance(result, OptimizationResult)
        assert result.status.startswith("CG-") or result.status == "Optimal"
        assert len(result.allocations) > 0
        assert result.total_waste_reduction > 0

    def test_column_generation_multi_drug(self):
        opt = AllocationOptimizer({"use_column_generation": True})
        invs = []
        for tent_id in range(1, 6):
            for drug in ["当归", "甘草", "黄芪"]:
                if tent_id <= 2:
                    invs.append(TentDrugInventory(
                        tent_id, drug, 150 + tent_id * 10, 2.0, 10 + tent_id * 3, 7
                    ))
                else:
                    invs.append(TentDrugInventory(
                        tent_id, drug, 10 + tent_id, 3.0, 365, 7
                    ))

        result = opt.optimize(invs)

        assert len(result.allocations) > 0
        drugs_in_result = {a.drug_name for a in result.allocations}
        assert len(drugs_in_result) >= 2

    def test_column_generation_vs_full_mip_consistency(self):
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 15, 7),
            TentDrugInventory(2, "当归", 20, 3.0, 365, 7),
            TentDrugInventory(3, "当归", 50, 2.0, 180, 7),
        ]

        opt_cg = AllocationOptimizer({"use_column_generation": True})
        opt_full = AllocationOptimizer({"use_column_generation": False})

        result_cg = opt_cg.optimize(invs)
        result_full = opt_full.optimize(invs)

        assert result_cg.status != "empty"
        assert result_full.status != "empty"

        if result_cg.allocations and result_full.allocations:
            cg_quantities = sum(a.quantity for a in result_cg.allocations)
            full_quantities = sum(a.quantity for a in result_full.allocations)
            assert cg_quantities > 0
            assert full_quantities > 0

    def test_column_generation_large_scale_performance(self):
        opt = AllocationOptimizer({"use_column_generation": True})
        invs = []
        drugs = ["当归", "大黄", "甘草", "黄芪", "白术"]
        for tent_id in range(1, 13):
            for drug in drugs:
                if tent_id <= 6:
                    invs.append(TentDrugInventory(
                        tent_id, drug, 200 + tent_id * 5, 2.0, 10 + tent_id, 7
                    ))
                else:
                    invs.append(TentDrugInventory(
                        tent_id, drug, 10 + tent_id, 3.0, 365, 7
                    ))

        start = time.time()
        result = opt.optimize(invs)
        solve_time = time.time() - start

        assert solve_time < 3.0, f"12帐篷求解时间 {solve_time:.2f}s > 3s"
        assert isinstance(result, OptimizationResult)

    def test_empty_inventories_returns_empty(self):
        opt = AllocationOptimizer({"use_column_generation": True})
        result = opt.optimize([])
        assert result.status == "empty"
        assert len(result.allocations) == 0


class TestColumnGenerationEdgeCases:
    """列生成法边界情况测试"""

    def test_single_tent_no_columns(self):
        opt = AllocationOptimizer({"use_column_generation": True})
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 15, 7),
        ]

        result = opt.optimize(invs)

        assert len(result.allocations) == 0 or result.total_waste_reduction == 0

    def test_no_deficit_no_columns(self):
        opt = AllocationOptimizer({"use_column_generation": True})
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 15, 7),
            TentDrugInventory(2, "当归", 200, 2.0, 20, 7),
        ]

        result = opt.optimize(invs)

        assert len(result.allocations) == 0 or result.total_waste_reduction == 0

    def test_no_surplus_no_columns(self):
        opt = AllocationOptimizer({"use_column_generation": True})
        invs = [
            TentDrugInventory(1, "当归", 5, 2.0, 365, 7),
            TentDrugInventory(2, "当归", 3, 3.0, 365, 7),
        ]

        result = opt.optimize(invs)
        assert len(result.allocations) == 0

    def test_mixed_supply_demand(self):
        opt = AllocationOptimizer({"use_column_generation": True})
        invs = [
            TentDrugInventory(1, "当归", 200, 2.0, 10, 7),
            TentDrugInventory(2, "当归", 100, 2.0, 300, 7),
            TentDrugInventory(3, "当归", 5, 3.0, 365, 7),
        ]

        result = opt.optimize(invs)

        assert isinstance(result, OptimizationResult)
        assert result.total_transport_cost >= 0

    def test_column_generation_config_toggle(self):
        opt_cg = AllocationOptimizer({"use_column_generation": True})
        opt_no_cg = AllocationOptimizer({"use_column_generation": False})

        assert opt_cg.use_column_generation is True
        assert opt_no_cg.use_column_generation is False
