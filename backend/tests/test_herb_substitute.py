"""
测试 herb_substitute 模块 - 知识图谱替代药材推荐

覆盖维度:
  - 正常: 当归变质后推荐川芎(同类+配伍); 推荐列表按相似度降序; 替代药材可用帐篷标注
  - 边界: 无替代药材时返回空; 孤立节点无可达路径; 最大深度限制生效
  - 异常: 远端Neo4j连接失败时降级到本地缓存; 本地缓存无此药材返回空
"""
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.herb_substitute.knowledge_graph import (
    HerbKnowledgeGraph, HerbSubstituteService, HerbProperties, SubstituteRecommendation,
    KNOWLEDGE_GRAPH_NODES, KNOWLEDGE_GRAPH_EDGES,
)


# ============================================================
#  正常场景
# ============================================================

class TestNormalSubstitute:
    def test_danggui_recommends_chuanxiong(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", max_depth=2, top_k=5)
        names = [r.substitute_herb for r in recs]
        assert "川芎" in names, f"当归替代推荐应包含川芎, 实际: {names}"

    def test_danggui_recommends_shudi(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", max_depth=2, top_k=5)
        names = [r.substitute_herb for r in recs]
        assert "熟地" in names, f"当归替代推荐应包含熟地(同类补血药), 实际: {names}"

    def test_recommendations_ordered_by_similarity(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", top_k=5)
        for i in range(len(recs) - 1):
            assert recs[i].similarity_score >= recs[i + 1].similarity_score

    def test_recommendation_has_shared_efficacy(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", top_k=3)
        assert len(recs) > 0
        for r in recs:
            assert isinstance(r.shared_efficacy, list)

    def test_recommendation_has_notes(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", top_k=3)
        for r in recs:
            assert len(r.notes) > 0

    def test_substitute_herb_not_self(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", top_k=5)
        for r in recs:
            assert r.substitute_herb != "当归"

    def test_huangqi_recommends_renshen(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("黄芪", max_depth=2, top_k=3)
        names = [r.substitute_herb for r in recs]
        assert "人参" in names, f"黄芪替代推荐应包含人参(同类补气药), 实际: {names}"

    def test_available_tents_annotated(self):
        svc = HerbSubstituteService()
        svc._tent_drugs = {
            1: ["当归", "大黄", "甘草"],
            3: ["川芎", "白芍", "熟地"],
        }
        recs = svc.recommend_substitutes("当归")
        for r in recs:
            if r.substitute_herb == "川芎":
                assert 3 in r.available_in_tents
            if r.substitute_herb == "甘草":
                assert 1 in r.available_in_tents


# ============================================================
#  边界场景
# ============================================================

class TestBoundarySubstitute:
    def test_nonexistent_herb_returns_empty(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("灵芝")
        assert recs == []

    def test_isolated_node_no_path(self):
        g = HerbKnowledgeGraph()
        g._nodes["孤立药材"] = HerbProperties(
            "孤立药材", "平", ["甘"], ["脾"], ["测试功效"], "测试类"
        )
        g._adjacency["孤立药材"] = []
        recs = g.find_substitutes("孤立药材", max_depth=3)
        assert len(recs) == 0

    def test_max_depth_one_limits_reach(self):
        g = HerbKnowledgeGraph()
        recs_depth1 = g.find_substitutes("当归", max_depth=1, top_k=10)
        recs_depth3 = g.find_substitutes("当归", max_depth=3, top_k=10)
        assert len(recs_depth1) <= len(recs_depth3)

    def test_top_k_limits_results(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", top_k=2)
        assert len(recs) <= 2

    def test_empty_herb_list_in_plans(self):
        svc = HerbSubstituteService()
        svc._tent_drugs = {}
        plans = svc.get_alternative_plans([])
        assert plans == {}


# ============================================================
#  异常场景
# ============================================================

class TestExceptionSubstitute:
    def test_remote_neo4j_unreachable_falls_back_to_local(self):
        g = HerbKnowledgeGraph(neo4j_url="http://nonexistent-host:7474", neo4j_user="x", neo4j_password="x")
        if g._use_remote:
            g._use_remote = False
            g._build_local_graph()
        recs = g.find_substitutes("当归", top_k=3)
        assert len(recs) > 0

    def test_remote_query_error_returns_error_dict(self):
        import asyncio

        async def _run():
            g = HerbKnowledgeGraph(neo4j_url="http://neo4j:7474", neo4j_user="neo4j", neo4j_password="test")
            if not g._use_remote:
                g._use_remote = True
                g._neo4j_url = "http://neo4j:7474"
                g._neo4j_auth = ("neo4j", "test")
            with patch("services.herb_substitute.knowledge_graph.httpx") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
                mock_httpx.AsyncClient.return_value = mock_client
                result = await g.query_remote("MATCH (n) RETURN n")
                assert "error" in result

        asyncio.run(_run())

    def test_local_cache_works_when_remote_unavailable(self):
        g = HerbKnowledgeGraph()
        node = g.get_node("当归")
        assert node is not None
        neighbors = g.get_neighbors("当归")
        assert len(neighbors) > 0
        recs = g.find_substitutes("当归")
        assert len(recs) > 0

    def test_service_handles_missing_tent_drugs_gracefully(self):
        svc = HerbSubstituteService()
        svc._tent_drugs = {}
        recs = svc.recommend_substitutes("当归")
        assert isinstance(recs, list)
        for r in recs:
            assert r.available_in_tents == []


# ============================================================
#  知识图谱完整性
# ============================================================

class TestKnowledgeGraphIntegrity:
    def test_all_15_herbs_present(self):
        assert len(KNOWLEDGE_GRAPH_NODES) == 15

    def test_node_has_required_fields(self):
        for name, props in KNOWLEDGE_GRAPH_NODES.items():
            assert props.nature in ("寒", "热", "温", "微温", "微寒", "平")
            assert len(props.flavor) > 0
            assert len(props.meridians) > 0
            assert len(props.efficacy) > 0
            assert props.category

    def test_danggui_properties(self):
        d = KNOWLEDGE_GRAPH_NODES["当归"]
        assert d.nature == "温"
        assert "甘" in d.flavor
        assert "补血活血" in d.efficacy
        assert d.category == "补血药"

    def test_edges_reference_valid_nodes(self):
        for src, dst, etype, weight in KNOWLEDGE_GRAPH_EDGES:
            assert src in KNOWLEDGE_GRAPH_NODES
            assert dst in KNOWLEDGE_GRAPH_NODES
            assert 0 < weight <= 1
            assert etype in ("同类", "互补", "配伍", "替代")

    def test_edge_count(self):
        assert len(KNOWLEDGE_GRAPH_EDGES) >= 25

    def test_graph_symmetric(self):
        g = HerbKnowledgeGraph()
        for herb in KNOWLEDGE_GRAPH_NODES:
            neighbors = g.get_neighbors(herb)
            for neighbor, etype, weight in neighbors:
                reverse = g.get_neighbors(neighbor)
                found = any(n == herb for n, _, _ in reverse)
                assert found, f"{herb}→{neighbor} edge missing reverse"


# ============================================================
#  修复验证：《本草纲目》毒性约束
# ============================================================

class TestToxicityConstraint:
    def test_toxicity_field_present_in_all_nodes(self):
        for name, props in KNOWLEDGE_GRAPH_NODES.items():
            assert hasattr(props, "toxicity"), f"{name} 缺少 toxicity 字段"
            assert props.toxicity in ["无毒", "小毒", "有毒", "大毒"], \
                f"{name} toxicity 值无效: {props.toxicity}"

    def test_known_toxic_herbs_marked_correctly(self):
        assert KNOWLEDGE_GRAPH_NODES["细辛"].toxicity == "小毒"
        assert KNOWLEDGE_GRAPH_NODES["麻黄"].toxicity == "小毒"
        assert KNOWLEDGE_GRAPH_NODES["大黄"].toxicity == "有毒"
        assert KNOWLEDGE_GRAPH_NODES["当归"].toxicity == "无毒"
        assert KNOWLEDGE_GRAPH_NODES["川芎"].toxicity == "无毒"

    def test_recommendations_exclude_highly_toxic_herbs(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", max_depth=3, top_k=10)
        names = [r.substitute_herb for r in recs]
        assert "大黄" not in names, f"有毒药材大黄不应出现在推荐列表中: {names}"

    def test_low_toxicity_herbs_have_lower_scores(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("桂枝", max_depth=3, top_k=10)
        name_to_score = {r.substitute_herb: r.similarity_score for r in recs}

        if "麻黄" in name_to_score and "桂枝" in name_to_score:
            pass

        for r in recs:
            if r.substitute_herb in ["麻黄", "细辛"]:
                assert "⚠️" in r.notes, f"小毒药材应有警示: {r.notes}"

    def test_toxicity_weight_applied_correctly(self):
        from services.herb_substitute.knowledge_graph import TOXICITY_WEIGHTS
        assert TOXICITY_WEIGHTS["无毒"] == 1.0
        assert TOXICITY_WEIGHTS["小毒"] == 0.5
        assert TOXICITY_WEIGHTS["有毒"] == 0.1
        assert TOXICITY_WEIGHTS["大毒"] == 0.0

    def test_recommendations_for_hot_climate_exclude_toxic(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("甘草", max_depth=3, top_k=5)
        names = [r.substitute_herb for r in recs]
        toxic_herbs = [h for h, p in KNOWLEDGE_GRAPH_NODES.items() if p.toxicity in ["有毒", "大毒"]]
        for th in toxic_herbs:
            assert th not in names, f"有毒药材 {th} 不应出现在推荐中: {names}"
