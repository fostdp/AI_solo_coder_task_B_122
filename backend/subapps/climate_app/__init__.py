"""
微气候调控子应用 - climate_app
独立FastAPI子应用，负责微气候调控策略评估
"""
from .main import app as climate_app

__all__ = ["climate_app"]
