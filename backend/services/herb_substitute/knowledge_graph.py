"""
Herb Substitute - 古代方剂替代药材推荐模块

基于《千金方》知识图谱 (模拟 Neo4j 图数据库),
通过图遍历推荐变质药材的替代品。

图模型:
  节点: 药材 (含性味归经、功效属性)
  边:   相似关系 (同类/互补/配伍/替代)

当某味药材变质不可用时:
  1. 从知识图谱中查找该药材节点
  2. 沿 替代/同类/互补 边遍历
  3. 按路径权重排序推荐替代药材
  4. 验证替代药材在当前帐篷的可用性
"""
import logging
import math
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HerbProperties:
    name: str
    nature: str
    flavor: List[str]
    meridians: List[str]
    efficacy: List[str]
    category: str


@dataclass
class SubstituteRecommendation:
    original_herb: str
    substitute_herb: str
    similarity_score: float
    path_type: str
    path_length: int
    shared_efficacy: List[str]
    shared_meridians: List[str]
    notes: str
    available_in_tents: List[int] = field(default_factory=list)


KNOWLEDGE_GRAPH_NODES: Dict[str, HerbProperties] = {
    "当归": HerbProperties("当归", "温", ["甘", "辛"], ["肝", "心", "脾"], ["补血活血", "调经止痛", "润肠通便"], "补血药"),
    "大黄": HerbProperties("大黄", "寒", ["苦"], ["脾", "胃", "大肠", "肝", "心"], ["泻下攻积", "清热泻火", "凉血解毒", "逐瘀通经"], "泻下药"),
    "甘草": HerbProperties("甘草", "平", ["甘"], ["心", "肺", "脾", "胃"], ["补脾益气", "清热解毒", "祛痰止咳", "缓急止痛", "调和诸药"], "补气药"),
    "黄芪": HerbProperties("黄芪", "微温", ["甘"], ["脾", "肺"], ["补气升阳", "固表止汗", "利水消肿", "生津养血"], "补气药"),
    "白术": HerbProperties("白术", "温", ["苦", "甘"], ["脾", "胃"], ["健脾益气", "燥湿利水", "止汗", "安胎"], "补气药"),
    "茯苓": HerbProperties("茯苓", "平", ["甘", "淡"], ["心", "肺", "脾", "肾"], ["利水渗湿", "健脾", "宁心"], "利水渗湿药"),
    "川芎": HerbProperties("川芎", "温", ["辛"], ["肝", "胆", "心包"], ["活血行气", "祛风止痛"], "活血化瘀药"),
    "白芍": HerbProperties("白芍", "微寒", ["苦", "酸"], ["肝", "脾"], ["养血调经", "敛阴止汗", "柔肝止痛", "平抑肝阳"], "补血药"),
    "熟地": HerbProperties("熟地", "微温", ["甘"], ["肝", "肾"], ["补血滋阴", "益精填髓"], "补血药"),
    "桂枝": HerbProperties("桂枝", "温", ["辛", "甘"], ["心", "肺", "膀胱"], ["发汗解肌", "温通经脉", "助阳化气"], "解表药"),
    "麻黄": HerbProperties("麻黄", "温", ["辛", "微苦"], ["肺", "膀胱"], ["发汗解表", "宣肺平喘", "利水消肿"], "解表药"),
    "细辛": HerbProperties("细辛", "温", ["辛"], ["肺", "肾"], ["祛风散寒", "通窍止痛", "温肺化饮"], "解表药"),
    "人参": HerbProperties("人参", "微温", ["甘", "微苦"], ["脾", "肺", "心", "肾"], ["大补元气", "补脾益肺", "生津养血", "安神益智"], "补气药"),
    "丹参": HerbProperties("丹参", "微寒", ["苦"], ["心", "心包", "肝"], ["活血祛瘀", "通经止痛", "清心除烦", "凉血消痈"], "活血化瘀药"),
    "五味子": HerbProperties("五味子", "温", ["酸", "甘"], ["肺", "心", "肾"], ["收敛固涩", "益气生津", "补肾宁心"], "收涩药"),
}

EDGE_TYPES = {
    "同类": 0.8,
    "互补": 0.6,
    "配伍": 0.5,
    "替代": 0.9,
}

KNOWLEDGE_GRAPH_EDGES: List[Tuple[str, str, str, float]] = [
    ("当归", "熟地", "同类", 0.85),
    ("当归", "白芍", "同类", 0.75),
    ("当归", "川芎", "配伍", 0.7),
    ("当归", "丹参", "同类", 0.65),
    ("黄芪", "人参", "同类", 0.85),
    ("黄芪", "白术", "同类", 0.7),
    ("黄芪", "甘草", "同类", 0.6),
    ("白术", "茯苓", "配伍", 0.8),
    ("白术", "黄芪", "同类", 0.7),
    ("茯苓", "白术", "配伍", 0.8),
    ("川芎", "丹参", "同类", 0.75),
    ("川芎", "当归", "配伍", 0.7),
    ("白芍", "当归", "同类", 0.75),
    ("白芍", "熟地", "配伍", 0.65),
    ("熟地", "当归", "同类", 0.85),
    ("熟地", "白芍", "配伍", 0.65),
    ("桂枝", "麻黄", "同类", 0.7),
    ("桂枝", "细辛", "配伍", 0.6),
    ("麻黄", "桂枝", "同类", 0.7),
    ("麻黄", "细辛", "配伍", 0.65),
    ("细辛", "桂枝", "配伍", 0.6),
    ("细辛", "麻黄", "配伍", 0.65),
    ("人参", "黄芪", "同类", 0.85),
    ("人参", "甘草", "同类", 0.55),
    ("丹参", "川芎", "同类", 0.75),
    ("丹参", "当归", "同类", 0.6),
    ("五味子", "人参", "互补", 0.55),
    ("甘草", "黄芪", "同类", 0.6),
    ("甘草", "白术", "同类", 0.5),
    ("大黄", "丹参", "互补", 0.45),
    ("大黄", "桂枝", "互补", 0.4),
]


class HerbKnowledgeGraph:
    """千金方知识图谱 - 模拟 Neo4j 图数据库"""

    def __init__(
        self,
        neo4j_url: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        self._adjacency: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        self._nodes = dict(KNOWLEDGE_GRAPH_NODES)
        self._use_remote = False

        if neo4j_url and HTTPX_AVAILABLE:
            self._neo4j_url = neo4j_url.rstrip("/")
            self._neo4j_auth = (neo4j_user or "", neo4j_password or "")
            self._use_remote = True
            logger.info("HerbKnowledgeGraph: remote Neo4j mode (%s)", neo4j_url)
        else:
            self._build_local_graph()
            logger.info("HerbKnowledgeGraph: local in-memory mode (15 herbs, %d edges)",
                        len(KNOWLEDGE_GRAPH_EDGES))

    def _build_local_graph(self):
        for src, dst, edge_type, weight in KNOWLEDGE_GRAPH_EDGES:
            self._adjacency[src].append((dst, edge_type, weight))
            self._adjacency[dst].append((src, edge_type, weight))

    def get_node(self, herb_name: str) -> Optional[HerbProperties]:
        return self._nodes.get(herb_name)

    def get_neighbors(self, herb_name: str) -> List[Tuple[str, str, float]]:
        return self._adjacency.get(herb_name, [])

    def find_substitutes(
        self,
        herb_name: str,
        max_depth: int = 3,
        top_k: int = 5,
        available_herbs: Optional[Set[str]] = None,
    ) -> List[SubstituteRecommendation]:
        node = self.get_node(herb_name)
        if node is None:
            return []

        candidates: Dict[str, Tuple[float, int, str, List[str]]] = {}
        visited = {herb_name}
        queue: List[Tuple[str, float, int, str, List[str]]] = [
            (herb_name, 1.0, 0, "start", [])
        ]

        while queue:
            current, cum_score, depth, path_type, path_herbs = queue.pop(0)
            if depth >= max_depth:
                continue

            for neighbor, edge_type, weight in self.get_neighbors(current):
                if neighbor in visited and neighbor != herb_name:
                    continue
                visited.add(neighbor)

                new_score = cum_score * weight * (0.8 ** depth)
                new_depth = depth + 1
                effective_path = path_type if depth == 0 else edge_type

                shared_eff = self._compute_shared_efficacy(herb_name, neighbor)
                eff_bonus = len(shared_eff) * 0.05
                new_score += eff_bonus

                neighbor_node = self.get_node(neighbor)
                if neighbor_node and neighbor != herb_name:
                    if neighbor not in candidates or candidates[neighbor][0] < new_score:
                        candidates[neighbor] = (
                            new_score, new_depth, effective_path, shared_eff
                        )

                if new_depth < max_depth:
                    queue.append((
                        neighbor, new_score, new_depth,
                        effective_path, path_herbs + [current]
                    ))

        results = []
        for herb, (score, depth, p_type, shared_eff) in sorted(
            candidates.items(), key=lambda x: -x[1][0]
        )[:top_k]:
            orig_node = self._nodes.get(herb_name)
            sub_node = self._nodes.get(herb)
            shared_mer = []
            if orig_node and sub_node:
                shared_mer = list(set(orig_node.meridians) & set(sub_node.meridians))

            notes = self._generate_notes(herb_name, herb, p_type, shared_eff)

            avail_tents = []
            if available_herbs:
                pass

            results.append(SubstituteRecommendation(
                original_herb=herb_name,
                substitute_herb=herb,
                similarity_score=round(min(1.0, score), 4),
                path_type=p_type,
                path_length=depth,
                shared_efficacy=shared_eff,
                shared_meridians=shared_mer,
                notes=notes,
                available_in_tents=avail_tents,
            ))

        return results

    def _compute_shared_efficacy(self, herb_a: str, herb_b: str) -> List[str]:
        node_a = self._nodes.get(herb_a)
        node_b = self._nodes.get(herb_b)
        if not node_a or not node_b:
            return []
        return list(set(node_a.efficacy) & set(node_b.efficacy))

    def _generate_notes(
        self, original: str, substitute: str, path_type: str, shared: List[str]
    ) -> str:
        orig_node = self._nodes.get(original)
        sub_node = self._nodes.get(substitute)
        parts = []

        if path_type == "同类":
            parts.append(f"与{original}同属{sub_node.category if sub_node else '未知'}类")
        elif path_type == "互补":
            parts.append(f"与{original}功效互补")
        elif path_type == "配伍":
            parts.append(f"常与{original}配伍使用")
        elif path_type == "替代":
            parts.append(f"临床常作为{original}的替代品")

        if shared:
            parts.append(f"共具{'+'.join(shared[:2])}之功")

        if orig_node and sub_node:
            if orig_node.nature != sub_node.nature:
                parts.append(f"注意: 药性从{orig_node.nature}变为{sub_node.nature}")

        return "；".join(parts) if parts else "可作替代"

    async def query_remote(self, cypher: str) -> dict:
        if not self._use_remote:
            return {"error": "No remote Neo4j configured"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{self._neo4j_url}/db/neo4j/tx/commit",
                    json={"statements": [{"statement": cypher}]},
                    headers={"Accept": "application/json"},
                    auth=self._neo4j_auth,
                )
                return resp.json()
        except Exception as e:
            logger.error("Neo4j query failed: %s", e)
            return {"error": str(e)}


class HerbSubstituteService:
    """药材替代推荐服务"""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._graph = HerbKnowledgeGraph(
            neo4j_url=cfg.get("neo4j_url"),
            neo4j_user=cfg.get("neo4j_user"),
            neo4j_password=cfg.get("neo4j_password"),
        )
        self._tent_drugs: Dict[int, List[str]] = {}

    async def start(self):
        from shared.config_loader import get_tents
        for tent in get_tents():
            self._tent_drugs[tent["id"]] = tent.get("drugs", [])
        logger.info("HerbSubstituteService started")

    async def stop(self):
        logger.info("HerbSubstituteService stopped")

    def recommend_substitutes(
        self,
        herb_name: str,
        max_depth: int = 3,
        top_k: int = 5,
    ) -> List[SubstituteRecommendation]:
        all_drugs = set()
        for drugs in self._tent_drugs.values():
            all_drugs.update(drugs)

        results = self._graph.find_substitutes(
            herb_name, max_depth=max_depth, top_k=top_k,
            available_herbs=all_drugs,
        )

        for rec in results:
            rec.available_in_tents = [
                tid for tid, drugs in self._tent_drugs.items()
                if rec.substitute_herb in drugs
            ]

        return results

    def get_alternative_plans(
        self, spoiled_herbs: List[dict],
    ) -> Dict[str, List[SubstituteRecommendation]]:
        plans = {}
        for item in spoiled_herbs:
            herb = item.get("drug_name") if isinstance(item, dict) else str(item)
            plans[herb] = self.recommend_substitutes(herb)
        return plans
