"""
方剂替代推荐子应用 - FastAPI主入口
独立子应用：/api/v3/herb
"""
from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List

from services.herb_substitute.knowledge_graph import HerbSubstituteService
from .schemas import HerbSubstituteRequest, SubstituteResponse, SubstituteItem

app = FastAPI(
    title="方剂替代推荐 API",
    description="基于《千金方》知识图谱的古代方剂替代推荐服务",
    version="3.0.0",
)

_herb_svc: Optional[HerbSubstituteService] = None


def init_service(service: HerbSubstituteService):
    global _herb_svc
    _herb_svc = service


@app.get("/health")
async def health():
    return {"status": "ok", "service": "substitute_app"}


@app.post("/substitutes", response_model=SubstituteResponse)
async def recommend_substitutes(req: HerbSubstituteRequest):
    if _herb_svc is None:
        raise HTTPException(status_code=503, detail="Herb substitute service not initialized")

    results = {}
    for herb_name in req.herb_names:
        recs = _herb_svc.recommend_substitutes(
            herb_name, max_depth=req.max_depth, top_k=req.top_k
        )
        results[herb_name] = [
            SubstituteItem(
                substitute_herb=r.substitute_herb,
                similarity_score=r.similarity_score,
                path_type=r.path_type,
                path_length=r.path_length,
                shared_efficacy=r.shared_efficacy,
                shared_meridians=r.shared_meridians,
                notes=r.notes,
                available_in_tents=r.available_in_tents,
            )
            for r in recs
        ]

    return SubstituteResponse(
        original_herbs=req.herb_names,
        recommendations=results,
    )


@app.get("/substitutes/{herb_name}")
async def get_single_substitute(
    herb_name: str,
    max_depth: int = Query(default=3, le=5),
    top_k: int = Query(default=5, le=10),
):
    if _herb_svc is None:
        raise HTTPException(status_code=503, detail="Herb substitute service not initialized")

    recs = _herb_svc.recommend_substitutes(herb_name, max_depth=max_depth, top_k=top_k)
    return {
        "original_herb": herb_name,
        "recommendations": [
            {
                "substitute_herb": r.substitute_herb,
                "similarity_score": r.similarity_score,
                "path_type": r.path_type,
                "path_length": r.path_length,
                "shared_efficacy": r.shared_efficacy,
                "shared_meridians": r.shared_meridians,
                "notes": r.notes,
                "available_in_tents": r.available_in_tents,
                "toxicity": _herb_svc._graph.get_node(r.substitute_herb).toxicity
                if _herb_svc._graph.get_node(r.substitute_herb) else None,
            }
            for r in recs
        ],
    }


@app.get("/graph/neighbors/{herb_name}")
async def get_herb_neighbors(herb_name: str):
    if _herb_svc is None:
        raise HTTPException(status_code=503, detail="Herb substitute service not initialized")

    graph = _herb_svc._graph
    node = graph.get_node(herb_name)
    if not node:
        raise HTTPException(status_code=404, detail=f"Herb '{herb_name}' not found in knowledge graph")

    neighbors = graph.get_neighbors(herb_name)
    return {
        "herb": {
            "name": node.name,
            "nature": node.nature,
            "flavor": node.flavor,
            "meridians": node.meridians,
            "efficacy": node.efficacy,
            "category": node.category,
            "toxicity": node.toxicity,
        },
        "neighbors": [
            {"name": n, "edge_type": et, "weight": w}
            for n, et, w in neighbors
        ],
    }


@app.get("/toxicity/{herb_name}")
async def get_herb_toxicity(herb_name: str):
    if _herb_svc is None:
        raise HTTPException(status_code=503, detail="Herb substitute service not initialized")

    node = _herb_svc._graph.get_node(herb_name)
    if not node:
        raise HTTPException(status_code=404, detail=f"Herb '{herb_name}' not found")

    return {
        "herb": herb_name,
        "toxicity": node.toxicity,
        "category": node.category,
        "safety_note": _get_safety_note(node.toxicity),
    }


@app.get("/herbs")
async def list_herbs(
    category: Optional[str] = None,
    toxicity: Optional[str] = None,
):
    if _herb_svc is None:
        raise HTTPException(status_code=503, detail="Herb substitute service not initialized")

    graph = _herb_svc._graph
    herbs = []
    for name, node in graph._nodes.items():
        if category and node.category != category:
            continue
        if toxicity and node.toxicity != toxicity:
            continue
        herbs.append({
            "name": name,
            "nature": node.nature,
            "category": node.category,
            "toxicity": node.toxicity,
            "efficacy_count": len(node.efficacy),
        })

    return {"total": len(herbs), "herbs": herbs}


def _get_safety_note(toxicity: str) -> str:
    notes = {
        "无毒": "安全常用药，可常规使用",
        "小毒": "有小毒，需遵医嘱，不宜过量久服",
        "有毒": "有毒，需严格控制用量，孕妇慎用",
        "大毒": "有大毒，一般外用，内服需极其谨慎",
    }
    return notes.get(toxicity, "安全性未知")
