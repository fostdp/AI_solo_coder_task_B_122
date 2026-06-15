"""
整数规划独立进程运行器
将PuLP求解放入独立进程，避免阻塞主事件循环
"""
import multiprocessing
import time
import pickle
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from .optimizer import (
    AllocationOptimizer, TentDrugInventory, OptimizationResult,
)

_executor: Optional[ProcessPoolExecutor] = None


def _ensure_executor():
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(
            max_workers=2,
            mp_context=multiprocessing.get_context("spawn"),
        )
    return _executor


def _solve_in_process(
    inventories_pickle: bytes,
    distances: Optional[Dict[Tuple[int, int], float]],
    config: Optional[dict] = None,
) -> OptimizationResult:
    """子进程中执行的求解函数"""
    inventories = pickle.loads(inventories_pickle)
    optimizer = AllocationOptimizer(config=config)
    return optimizer.optimize(inventories, distances)


async def optimize_async(
    inventories: List[TentDrugInventory],
    distances: Optional[Dict[Tuple[int, int], float]] = None,
    config: Optional[dict] = None,
    timeout: float = 10.0,
) -> OptimizationResult:
    """
    异步求解（独立进程）
    - 小规模问题：直接在当前线程求解
    - 大规模问题：提交到进程池
    """
    if len(inventories) < 20:
        optimizer = AllocationOptimizer(config=config)
        return optimizer.optimize(inventories, distances)

    try:
        import asyncio
        inv_pickle = pickle.dumps(inventories)
        executor = _ensure_executor()

        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor, _solve_in_process, inv_pickle, distances, config
            ),
            timeout=timeout,
        )
        return result

    except TimeoutError:
        optimizer = AllocationOptimizer(config=config)
        return optimizer._heuristic_optimize(inventories, distances)

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Process pool optimization failed: {e}, falling back to in-process")
        optimizer = AllocationOptimizer(config=config)
        return optimizer.optimize(inventories, distances)


def optimize_sync(
    inventories: List[TentDrugInventory],
    distances: Optional[Dict[Tuple[int, int], float]] = None,
    config: Optional[dict] = None,
    timeout: float = 10.0,
) -> OptimizationResult:
    """同步版本的独立进程求解"""
    if len(inventories) < 20:
        optimizer = AllocationOptimizer(config=config)
        return optimizer.optimize(inventories, distances)

    try:
        inv_pickle = pickle.dumps(inventories)
        executor = _ensure_executor()
        future = executor.submit(_solve_in_process, inv_pickle, distances, config)
        return future.result(timeout=timeout)

    except TimeoutError:
        optimizer = AllocationOptimizer(config=config)
        return optimizer._heuristic_optimize(inventories, distances)

    except Exception:
        optimizer = AllocationOptimizer(config=config)
        return optimizer.optimize(inventories, distances)


def shutdown():
    """关闭进程池"""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None
