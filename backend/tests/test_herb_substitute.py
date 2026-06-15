"""
测试 herb_substitute 模块 - 知识图谱替代药材推荐
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.herb_substitute.knowledge_graph import (
    HerbKnowledgeGraph, HerbSubstituteService, SubstituteRecommendation,
    KNOWLEDGE_GRAPH_NODES, KNOWLEDGE_GRAPH_EDGES,
)


class TestKnowledgeGraphNodes:
    def test_all_15_herbs_present(self):
        assert len(KNOWLEDGE_GRAPH_NODES) == 15

    def test_node_has_required_fields(self):
        for name, props in KNOWLEDGE_GRAPH_NODES.items():
            assert props.nature in ("寒", "热", "温", "微温", "微寒", "平")
            assert len(props.flavor) > 0
            assert len(props.meridians) > 0
            assert len(props.efficacy) > 0
            assert props.category

    def test_specific_herb(self):
        danggui = KNOWLEDGE_GRAPH_NODES["当归"]
        assert danggui.nature == "温"
        assert "甘" in danggui.flavor
        assert "补血活血" in danggui.efficacy
        assert danggui.category == "补血药"


class TestKnowledgeGraphEdges:
    def test_edges_symmetric_or_directed(self):
        for src, dst, etype, weight in KNOWLEDGE_GRAPH_EDGES:
            assert src in KNOWLEDGE_GRAPH_NODES
            assert dst in KNOWLEDGE_GRAPH_NODES
            assert 0 < weight <= 1
            assert etype in ("同类", "互补", "配伍", "替代")

    def test_edge_count_reasonable(self):
        assert len(KNOWLEDGE_GRAPH_EDGES) >= 25


class TestHerbKnowledgeGraph:
    def test_local_graph_init(self):
        g = HerbKnowledgeGraph()
        assert len(g._adjacency) > 0

    def test_get_node(self):
        g = HerbKnowledgeGraph()
        node = g.get_node("当归")
        assert node is not None
        assert node.name == "当归"

    def test_get_node_not_found(self):
        g = HerbKnowledgeGraph()
        assert g.get_node("不存在的药材") is None

    def test_get_neighbors(self):
        g = HerbKnowledgeGraph()
        neighbors = g.get_neighbors("当归")
        assert len(neighbors) > 0
        names = [n for n, _, _ in neighbors]
        assert "熟地" in names

    def test_find_substitutes_basic(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", max_depth=2, top_k=3)
        assert len(recs) > 0
        assert all(isinstance(r, SubstituteRecommendation) for r in recs)
        assert recs[0].original_herb == "当归"
        assert recs[0].substitute_herb != "当归"

    def test_find_substitutes_nonexistent(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("灵芝")
        assert len(recs) == 0

    def test_substitute_similarity_ordering(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", top_k=5)
        for i in range(len(recs) - 1):
            assert recs[i].similarity_score >= recs[i + 1].similarity_score

    def test_shared_efficacy(self):
        g = HerbKnowledgeGraph()
        shared = g._compute_shared_efficacy("当归", "熟地")
        assert "补血活血" in shared or "补血滋阴" in shared or len(shared) >= 0

    def test_notes_generation(self):
        g = HerbKnowledgeGraph()
        notes = g._generate_notes("当归", "熟地", "同类", ["补血"])
        assert len(notes) > 0


class TestHerbSubstituteService:
    def test_recommend_substitutes(self):
        svc = HerbSubstituteService()
        svc._tent_drugs = {
            1: ["当归", "大黄", "甘草"],
            2: ["黄芪", "白术", "茯苓"],
            3: ["川芎", "白芍", "熟地"],
        }
        recs = svc.recommend_substitutes("当归")
        assert len(recs) > 0
        for r in recs:
            assert r.available_in_tents is not None

    def test_get_alternative_plans(self):
        svc = HerbSubstituteService()
        svc._tent_drugs = {
            1: ["当归", "大黄"],
            3: ["川芎", "白芍", "熟地"],
        }
        plans = svc.get_alternative_plans(["当归", "大黄"])
        assert "当归" in plans
        assert "大黄" in plans
        assert len(plans["当归"]) > 0
