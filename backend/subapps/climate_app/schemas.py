"""
微气候调控子应用 - 请求/响应模型
"""
from pydantic import BaseModel
from typing import Optional, List


class ClimateControlRequest(BaseModel):
    tent_id: int
    temperature: float = 25.0
    humidity: float = 60.0
    light: float = 400.0
    aw: float = 0.50


class ActionInfo(BaseModel):
    ventilation: int
    shading: int
    humidifier: int
    action_index: int
    description: str
    energy_cost: float


class ClimateStateInfo(BaseModel):
    temperature: float
    humidity: float
    light: float
    aw: float


class ClimateControlResponse(BaseModel):
    tent_id: int
    action: ActionInfo
    expected_reward: float
    current_state: ClimateStateInfo
    projected_state: ClimateStateInfo
    shelf_life_improvement_days: float
    description: str
    algorithm: Optional[str] = None


class TrainingRequest(BaseModel):
    episodes: int = 200
    max_steps: int = 48
    use_potential_shaping: bool = True


class TrainingResponse(BaseModel):
    status: str
    episodes: int
    final_epsilon: float
    avg_reward_last_10: float
    message: str
