"""
方剂替代推荐子应用 - 请求/响应模型
"""
from pydantic import BaseModel
from typing import List, Optional


class HerbSubstituteRequest(BaseModel):
    herb_names: List[str]
    max_depth: int = 3
    top_k: int = 5
    include_toxic: bool = False


class SubstituteItem(BaseModel):
    substitute_herb: str
    similarity_score: float
    path_type: str
    path_length: int
    shared_efficacy: List[str]
    shared_meridians: List[str]
    notes: str
    available_in_tents: List[int]
    toxicity: Optional[str] = None


class SubstituteResponse(BaseModel):
    original_herbs: List[str]
    recommendations: dict
