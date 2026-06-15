"""
微气候调控子应用 - FastAPI主入口
独立子应用：/api/v3/climate
"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from typing import Optional, List

from services.climate_control.dqn_controller import (
    ClimateControlService, ControlAction,
)
from .schemas import (
    ClimateControlRequest, ClimateControlResponse,
    ActionInfo, ClimateStateInfo,
    TrainingRequest, TrainingResponse,
)

app = FastAPI(
    title="微气候调控 API",
    description="基于DQN强化学习的医疗帐篷微气候调控策略服务",
    version="3.0.0",
)

_climate_svc: Optional[ClimateControlService] = None
_training_task = None


def init_service(service: ClimateControlService):
    global _climate_svc
    _climate_svc = service


@app.get("/health")
async def health():
    return {"status": "ok", "service": "climate_app"}


@app.post("/recommend", response_model=ClimateControlResponse)
async def recommend_climate_control(req: ClimateControlRequest):
    if _climate_svc is None:
        raise HTTPException(status_code=503, detail="Climate control service not initialized")

    climate = {
        "temperature": req.temperature,
        "humidity": req.humidity,
        "light": req.light,
        "aw": req.aw,
    }
    rec = _climate_svc.recommend(req.tent_id, climate)

    return ClimateControlResponse(
        tent_id=rec.tent_id,
        action=ActionInfo(
            ventilation=rec.action.ventilation,
            shading=rec.action.shading,
            humidifier=rec.action.humidifier,
            action_index=rec.action.index,
            description=rec.action.describe(),
            energy_cost=rec.action.energy_cost(),
        ),
        expected_reward=rec.expected_reward,
        current_state=ClimateStateInfo(
            temperature=rec.current_state.temperature,
            humidity=rec.current_state.humidity,
            light=rec.current_state.light,
            aw=rec.current_state.aw,
        ),
        projected_state=ClimateStateInfo(
            temperature=rec.projected_state.temperature,
            humidity=rec.projected_state.humidity,
            light=rec.projected_state.light,
            aw=rec.projected_state.aw,
        ),
        shelf_life_improvement_days=rec.shelf_life_improvement_days,
        description=rec.description,
        algorithm="dqn" if rec.expected_reward > 0.05 else "rule_based",
    )


@app.post("/batch-recommend")
async def batch_climate_recommend():
    if _climate_svc is None:
        raise HTTPException(status_code=503, detail="Climate control service not initialized")

    from shared.config_loader import get_tents

    results = []
    for tent in get_tents():
        climate = {"temperature": 25, "humidity": 60, "light": 400, "aw": 0.50}
        rec = _climate_svc.recommend(tent["id"], climate)
        results.append({
            "tent_id": tent["id"],
            "tent_name": tent["name"],
            "action_description": rec.description,
            "shelf_life_improvement_days": rec.shelf_life_improvement_days,
            "expected_reward": rec.expected_reward,
            "algorithm": "dqn" if rec.expected_reward > 0.05 else "rule_based",
        })

    return {"recommendations": results}


@app.post("/train", response_model=TrainingResponse)
async def train_dqn(req: TrainingRequest, background_tasks: BackgroundTasks):
    if _climate_svc is None:
        raise HTTPException(status_code=503, detail="Climate control service not initialized")

    episodes = min(req.episodes, 1000)

    from services.climate_control.dqn_controller import TentClimateSimulator
    simulator = TentClimateSimulator({"temperature": 30, "humidity": 65, "light": 600, "aw": 0.55})

    result = _climate_svc._agent.train_episodes(
        simulator,
        episodes=episodes,
        max_steps=req.max_steps,
        use_potential_shaping=req.use_potential_shaping,
    )

    rewards = result["rewards"]
    avg_last_10 = sum(rewards[-10:]) / 10 if len(rewards) >= 10 else sum(rewards) / len(rewards)

    return TrainingResponse(
        status="ok",
        episodes=episodes,
        final_epsilon=round(_climate_svc._agent.epsilon, 4),
        avg_reward_last_10=round(avg_last_10, 4),
        message=f"DQN agent trained with {episodes} episodes, potential_shaping={req.use_potential_shaping}",
    )


@app.get("/actions")
async def list_actions():
    actions = []
    for i in range(len(ControlAction.ACTION_MAP)):
        action = ControlAction.from_index(i)
        actions.append({
            "index": i,
            "ventilation": action.ventilation,
            "shading": action.shading,
            "humidifier": action.humidifier,
            "description": action.describe(),
            "energy_cost": action.energy_cost(),
        })
    return {"total_actions": len(actions), "actions": actions}


@app.get("/agent/status")
async def get_agent_status():
    if _climate_svc is None:
        return {"status": "not_initialized"}
    agent = _climate_svc._agent
    return {
        "status": "ready",
        "epsilon": round(agent.epsilon, 4),
        "epsilon_end": agent.epsilon_end,
        "gamma": agent.gamma,
        "lr": agent.lr,
        "batch_size": agent.batch_size,
        "replay_buffer_size": len(agent.replay_buffer),
    }
