"""
调配优化子应用 - allocation_app
独立FastAPI子应用，负责药材调配优先级决策
"""
from .main import app as allocation_app

__all__ = ["allocation_app"]
