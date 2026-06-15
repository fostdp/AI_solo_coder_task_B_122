"""
毒性过滤器单元测试
测试《本草纲目》毒性约束的过滤逻辑和权重计算
"""
import pytest

from services.herb_substitute.knowledge_graph import (
    HerbKnowledgeGraph, KNOWLEDGE_GRAPH_NODES, TOXICITY_WEIGHTS,
    HerbProperties,
)


class TestToxicityWeights:
    """毒性权重映射测试"""

    def test_toxicity_weights_complete(self):
        assert "无毒" in TOXICITY_WEIGHTS
        assert "小毒" in TOXICITY_WEIGHTS
        assert "有毒" in TOXICITY_WEIGHTS
        assert "大毒" in TOXICITY_WEIGHTS

    def test_toxicity_weights_monotonic(self):
        assert TOXICITY_WEIGHTS["无毒"] == 1.0
        assert TOXICITY_WEIGHTS["小毒"] == 0.5
        assert TOXICITY_WEIGHTS["有毒"] == 0.1
        assert TOXICITY_WEIGHTS["大毒"] == 0.0

        assert TOXICITY_WEIGHTS["无毒"] > TOXICITY_WEIGHTS["小毒"]
        assert TOXICITY_WEIGHTS["小毒"] > TOXICITY_WEIGHTS["有毒"]
        assert TOXICITY_WEIGHTS["有毒"] > TOXICITY_WEIGHTS["大毒"]

    def test_toxicity_weights_range(self):
        for level, weight in TOXICITY_WEIGHTS.items():
            assert 0 <= weight <= 1.0, f"{level} 权重 {weight} 不在 [0,1] 范围内"


class TestNodeToxicityData:
    """药材节点毒性数据测试"""

    def test_all_nodes_have_toxicity(self):
        for name, props in KNOWLEDGE_GRAPH_NODES.items():
            assert hasattr(props, "toxicity"), f"{name} 缺少 toxicity 属性"
            assert isinstance(props.toxicity, str), f"{name} toxicity 应为字符串"

    def test_all_toxicity_values_valid(self):
        valid = set(TOXICITY_WEIGHTS.keys())
        for name, props in KNOWLEDGE_GRAPH_NODES.items():
            assert props.toxicity in valid, \
                f"{name} toxicity='{props.toxicity}' 不在有效值 {valid} 中"

    def test_known_toxic_herbs_correctly_labeled(self):
        assert KNOWLEDGE_GRAPH_NODES["细辛"].toxicity == "小毒"
        assert KNOWLEDGE_GRAPH_NODES["麻黄"].toxicity == "小毒"
        assert KNOWLEDGE_GRAPH_NODES["大黄"].toxicity == "有毒"
        assert KNOWLEDGE_GRAPH_NODES["当归"].toxicity == "无毒"
        assert KNOWLEDGE_GRAPH_NODES["甘草"].toxicity == "无毒"
        assert KNOWLEDGE_GRAPH_NODES["黄芪"].toxicity == "无毒"

    def test_toxicity_distribution(self):
        counts = {}
        for props in KNOWLEDGE_GRAPH_NODES.values():
            counts[props.toxicity] = counts.get(props.toxicity, 0) + 1

        assert counts.get("无毒", 0) > 5
        assert counts.get("小毒", 0) >= 2
        assert counts.get("有毒", 0) >= 1


class TestToxicityFilter:
    """毒性过滤逻辑测试"""

    def test_highly_toxic_excluded_from_recommendations(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", max_depth=3, top_k=10)
        names = [r.substitute_herb for r in recs]

        toxic_herbs = [
            h for h, p in KNOWLEDGE_GRAPH_NODES.items()
            if p.toxicity in ["有毒", "大毒"]
        ]

        for th in toxic_herbs:
            assert th not in names, f"有毒/大毒药材 {th} 不应出现在推荐中"

    def test_low_toxicity_can_appear_with_lower_score(self):
        g = HerbKnowledgeGraph()

        all_recs = g.find_substitutes("桂枝", max_depth=3, top_k=10)
        low_toxic_recs = [
            r for r in all_recs
            if KNOWLEDGE_GRAPH_NODES.get(r.substitute_herb, {}).__dict__.get("toxicity") == "小毒"
        ]
        non_toxic_recs = [
            r for r in all_recs
            if KNOWLEDGE_GRAPH_NODES.get(r.substitute_herb, {}).__dict__.get("toxicity") == "无毒"
        ]

        if low_toxic_recs and non_toxic_recs:
            max_low_score = max(r.similarity_score for r in low_toxic_recs)
            max_non_toxic_score = max(r.similarity_score for r in non_toxic_recs)
            assert max_low_score <= max_non_toxic_score * 1.5

    def test_danggui_recommendations_all_safe(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", max_depth=3, top_k=5)

        for r in recs:
            node = g.get_node(r.substitute_herb)
            assert node is not None
            assert node.toxicity not in ["有毒", "大毒"], \
                f"{r.substitute_herb}({node.toxicity}) 不应被推荐"

    def test_toxicity_applied_in_bfs(self):
        g = HerbKnowledgeGraph()

        recs_depth1 = g.find_substitutes("当归", max_depth=1, top_k=10)
        recs_depth3 = g.find_substitutes("当归", max_depth=3, top_k=10)

        for r in recs_depth1:
            node = g.get_node(r.substitute_herb)
            assert node.toxicity not in ["有毒", "大毒"]

        for r in recs_depth3:
            node = g.get_node(r.substitute_herb)
            assert node.toxicity not in ["有毒", "大毒"]

    def test_isolated_toxic_node_no_path(self):
        g = HerbKnowledgeGraph()

        recs = g.find_substitutes("大黄", max_depth=3, top_k=10)
        assert isinstance(recs, list)

        names = [r.substitute_herb for r in recs]
        for n in names:
            node = g.get_node(n)
            assert node.toxicity not in ["有毒", "大毒"]


class TestToxicityNotes:
    """毒性警示语测试"""

    def test_toxic_herbs_have_warning_notes(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("桂枝", max_depth=3, top_k=10)

        for r in recs:
            node = g.get_node(r.substitute_herb)
            if node and node.toxicity in ["小毒", "有毒", "大毒"]:
                assert "⚠️" in r.notes or "毒" in r.notes, \
                    f"{r.substitute_herb} 应有毒性警示: {r.notes}"

    def test_non_toxic_no_warning_prefix(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", max_depth=2, top_k=3)

        non_toxic_recs = [
            r for r in recs
            if g.get_node(r.substitute_herb).toxicity == "无毒"
        ]

        if non_toxic_recs:
            r = non_toxic_recs[0]
            assert not r.notes.startswith("⚠️"), \
                f"无毒药材不应有⚠️前缀: {r.notes}"

    def test_generate_notes_with_toxicity(self):
        g = HerbKnowledgeGraph()

        note_safe = g._generate_notes("当归", "川芎", "同类", ["活血"], "无毒")
        assert "⚠️" not in note_safe

        note_toxic = g._generate_notes("当归", "麻黄", "同类", ["解表"], "小毒")
        assert "小毒" in note_toxic
        assert "⚠️" in note_toxic

        note_highly_toxic = g._generate_notes("当归", "大黄", "同类", ["泻下"], "有毒")
        assert "有毒" in note_highly_toxic
        assert "⚠️" in note_highly_toxic


class TestToxicityServiceIntegration:
    """毒性过滤器与服务层集成测试"""

    def test_service_recommendations_exclude_highly_toxic(self):
        from services.herb_substitute.knowledge_graph import HerbSubstituteService

        svc = HerbSubstituteService()
        recs = svc.recommend_substitutes("当归", max_depth=3, top_k=10)

        for r in recs:
            node = svc._graph.get_node(r.substitute_herb)
            assert node.toxicity not in ["有毒", "大毒"]

    def test_service_with_multi_herbs(self):
        from services.herb_substitute.knowledge_graph import HerbSubstituteService

        svc = HerbSubstituteService()
        herbs_to_test = ["当归", "甘草", "黄芪", "桂枝"]

        for herb in herbs_to_test:
            recs = svc.recommend_substitutes(herb, max_depth=3, top_k=5)
            for r in recs:
                node = svc._graph.get_node(r.substitute_herb)
                assert node.toxicity not in ["有毒", "大毒"], \
                    f"{herb} 的推荐 {r.substitute_herb}({node.toxicity}) 不应包含有毒药材"

    def test_toxicity_penalty_reduces_score(self):
        g = HerbKnowledgeGraph()

        recs = g.find_substitutes("桂枝", max_depth=2, top_k=10)

        low_toxic = [r for r in recs
                     if g.get_node(r.substitute_herb).toxicity == "小毒"]
        non_toxic = [r for r in recs
                     if g.get_node(r.substitute_herb).toxicity == "无毒"]

        if low_toxic and non_toxic:
            avg_low = sum(r.similarity_score for r in low_toxic) / len(low_toxic)
            avg_non = sum(r.similarity_score for r in non_toxic) / len(non_toxic)

            assert avg_low <= avg_non * 1.2, \
                f"小毒药材平均分 {avg_low:.3f} 不应显著高于无毒 {avg_non:.3f}"


class TestToxicityEdgeCases:
    """毒性过滤器边界测试"""

    def test_empty_graph_no_crash(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("不存在的药材", max_depth=3, top_k=5)
        assert recs == []

    def test_self_not_included(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", max_depth=3, top_k=10)
        names = [r.substitute_herb for r in recs]
        assert "当归" not in names

    def test_max_depth_zero_returns_empty(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", max_depth=0, top_k=5)
        assert recs == []

    def test_top_k_limits_results(self):
        g = HerbKnowledgeGraph()
        recs = g.find_substitutes("当归", max_depth=3, top_k=3)
        assert len(recs) <= 3

    def test_all_toxic_node_no_recommendations(self):
        g = HerbKnowledgeGraph()

        recs = g.find_substitutes("大黄", max_depth=1, top_k=5)
        assert isinstance(recs, list)

        for r in recs:
            node = g.get_node(r.substitute_herb)
            assert node.toxicity in ["无毒", "小毒"], \
                f"推荐 {r.substitute_herb} 毒性 {node.toxicity} 过高"
