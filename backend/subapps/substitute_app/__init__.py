"""
方剂替代推荐子应用 - substitute_app
独立FastAPI子应用，负责古代方剂替代推荐
"""
from .main import app as substitute_app

__all__ = ["substitute_app"]
