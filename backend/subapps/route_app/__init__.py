"""
路径影响分析子应用 - route_app
独立FastAPI子应用，负责商队路径规划影响分析
"""
from .main import app as route_app

__all__ = ["route_app"]
