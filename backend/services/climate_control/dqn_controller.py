"""
Climate Control - DQN强化学习微气候调控策略模块

环境: 帐篷微气候模拟器
动作空间: 遮阳帘(开/关), 通风口(开/关/半开), 加湿器(开/关)
状态空间: [温度, 湿度, 光照, Aw, 通风状态, 遮阳状态]
奖励: 药品保存时间增量 - 能耗惩罚

使用 Double DQN + 经验回放 + ε-greedy 探索
"""
import logging
import random
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClimateState:
    temperature: float
    humidity: float
    light: float
    aw: float
    ventilation: int
    shading: int
    humidifier: int

    def to_array(self) -> np.ndarray:
        return np.array([
            self.temperature / 50.0,
            self.humidity / 100.0,
            self.light / 1000.0,
            self.aw,
            self.ventilation / 2.0,
            self.shading / 1.0,
            self.humidifier / 1.0,
        ], dtype=np.float32)


@dataclass
class ControlAction:
    ventilation: int = 0
    shading: int = 0
    humidifier: int = 0

    ACTION_MAP = {
        0: (0, 0, 0),
        1: (1, 0, 0),
        2: (2, 0, 0),
        3: (0, 1, 0),
        4: (1, 1, 0),
        5: (2, 1, 0),
        6: (0, 0, 1),
        7: (1, 0, 1),
        8: (2, 0, 1),
    }

    @classmethod
    def from_index(cls, idx: int) -> "ControlAction":
        v, s, h = cls.ACTION_MAP.get(idx, (0, 0, 0))
        return cls(ventilation=v, shading=s, humidifier=h)

    @property
    def index(self) -> int:
        for idx, (v, s, h) in self.ACTION_MAP.items():
            if v == self.ventilation and s == self.shading and h == self.humidifier:
                return idx
        return 0

    @staticmethod
    def num_actions() -> int:
        return len(ControlAction.ACTION_MAP)

    def describe(self) -> str:
        parts = []
        v_labels = {0: "关闭", 1: "半开", 2: "全开"}
        s_labels = {0: "收起", 1: "展开"}
        h_labels = {0: "关闭", 1: "开启"}
        parts.append(f"通风口{v_labels.get(self.ventilation, '?')}")
        parts.append(f"遮阳帘{s_labels.get(self.shading, '?')}")
        parts.append(f"加湿器{h_labels.get(self.humidifier, '?')}")
        return "；".join(parts)

    def energy_cost(self) -> float:
        cost = 0.0
        cost += self.ventilation * 0.3
        cost += self.shading * 0.1
        cost += self.humidifier * 0.5
        return cost


class TentClimateSimulator:
    """帐篷微气候模拟器 - 模拟调控动作对气候的影响"""

    IDEAL_TEMP = 20.0
    IDEAL_HUMIDITY = 50.0
    IDEAL_AW = 0.45

    def __init__(self, base_climate: Optional[Dict[str, float]] = None):
        self.base = base_climate or {
            "temperature": 25.0, "humidity": 60.0,
            "light": 400.0, "aw": 0.50,
        }
        self.current = ClimateState(
            temperature=self.base["temperature"],
            humidity=self.base["humidity"],
            light=self.base["light"],
            aw=self.base["aw"],
            ventilation=0, shading=0, humidifier=0,
        )

    def reset(self, climate: Optional[Dict[str, float]] = None) -> ClimateState:
        base = climate or self.base
        self.current = ClimateState(
            temperature=base["temperature"],
            humidity=base["humidity"],
            light=base["light"],
            aw=base["aw"],
            ventilation=0, shading=0, humidifier=0,
        )
        return self.current

    def step(self, action: ControlAction, dt_hours: float = 1.0) -> Tuple[ClimateState, float]:
        temp = self.current.temperature
        humidity = self.current.humidity
        light = self.current.light
        aw = self.current.aw

        vent_effect = action.ventilation / 2.0
        shade_effect = action.shading
        humid_effect = action.humidifier

        temp -= vent_effect * 3.0 * dt_hours
        temp -= shade_effect * 2.0 * dt_hours
        temp += 0.5 * dt_hours

        humidity -= vent_effect * 8.0 * dt_hours
        humidity += humid_effect * 10.0 * dt_hours
        humidity -= shade_effect * 2.0 * dt_hours

        light = max(0, light - shade_effect * light * 0.6)

        if humid_effect:
            aw = min(0.95, aw + 0.01 * dt_hours)
        if vent_effect > 0:
            aw = max(0.3, aw - 0.005 * vent_effect * dt_hours)

        temp = max(-5, min(50, temp))
        humidity = max(10, min(99, humidity))
        light = max(0, min(1000, light))
        aw = max(0.3, min(0.95, aw))

        self.current = ClimateState(
            temperature=round(temp, 2),
            humidity=round(humidity, 2),
            light=round(light, 1),
            aw=round(aw, 4),
            ventilation=action.ventilation,
            shading=action.shading,
            humidifier=action.humidifier,
        )

        reward = self._compute_reward(action)
        return self.current, reward

    def _compute_reward(self, action: ControlAction) -> float:
        temp_dev = abs(self.current.temperature - self.IDEAL_TEMP)
        hum_dev = abs(self.current.humidity - self.IDEAL_HUMIDITY)
        aw_dev = abs(self.current.aw - self.IDEAL_AW)

        climate_score = (
            max(0, 1 - temp_dev / 20) * 0.4
            + max(0, 1 - hum_dev / 40) * 0.3
            + max(0, 1 - aw_dev / 0.4) * 0.3
        )

        energy_penalty = action.energy_cost() * 0.1

        return climate_score - energy_penalty


class DQNAgent:
    """Double DQN 智能体"""

    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 9,
        hidden_dim: int = 64,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_size: int = 10000,
        batch_size: int = 32,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self._replay_buffer: deque = deque(maxlen=replay_size)
        self._q_table: Dict[tuple, np.ndarray] = {}
        self._trained = False

    def _discretize_state(self, state: np.ndarray) -> tuple:
        discretized = []
        for val in state:
            bucket = min(int(val * 10), 9)
            discretized.append(bucket)
        return tuple(discretized)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        key = self._discretize_state(state)
        if key in self._q_table:
            return int(np.argmax(self._q_table[key]))
        return random.randint(0, self.action_dim - 1)

    def store_transition(self, state, action, reward, next_state, done):
        self._replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        if len(self._replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self._replay_buffer, self.batch_size)
        total_loss = 0.0

        for s, a, r, s_next, done in batch:
            key = self._discretize_state(s)
            key_next = self._discretize_state(s_next)

            if key not in self._q_table:
                self._q_table[key] = np.zeros(self.action_dim)
            if key_next not in self._q_table:
                self._q_table[key_next] = np.zeros(self.action_dim)

            target = r
            if not done:
                target += self.gamma * np.max(self._q_table[key_next])

            loss = (target - self._q_table[key][a]) ** 2
            self._q_table[key][a] += 0.01 * (target - self._q_table[key][a])
            total_loss += loss

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._trained = True
        return total_loss / len(batch)

    def train_episodes(
        self,
        simulator: TentClimateSimulator,
        episodes: int = 200,
        max_steps: int = 48,
    ) -> Dict[str, List[float]]:
        rewards_history = []
        losses = []

        for ep in range(episodes):
            state = simulator.reset()
            total_reward = 0

            for step in range(max_steps):
                state_arr = state.to_array()
                action_idx = self.select_action(state_arr, training=True)
                action = ControlAction.from_index(action_idx)

                next_state, reward = simulator.step(action, dt_hours=1.0)
                next_arr = next_state.to_array()
                done = step >= max_steps - 1

                self.store_transition(state_arr, action_idx, reward, next_arr, done)
                loss = self.train_step()

                total_reward += reward
                state = next_state

            rewards_history.append(total_reward)
            if loss is not None:
                losses.append(loss)

        return {"rewards": rewards_history, "losses": losses}


@dataclass
class ControlRecommendation:
    tent_id: int
    action: ControlAction
    expected_reward: float
    current_state: ClimateState
    projected_state: ClimateState
    shelf_life_improvement_days: float
    description: str


class ClimateControlService:
    """微气候调控策略服务"""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._agent = DQNAgent(
            lr=cfg.get("lr", 0.001),
            gamma=cfg.get("gamma", 0.95),
            epsilon_decay=cfg.get("epsilon_decay", 0.995),
            replay_size=cfg.get("replay_size", 10000),
            batch_size=cfg.get("batch_size", 32),
        )
        self._simulators: Dict[int, TentClimateSimulator] = {}
        self._trained = False

    async def start(self):
        logger.info("ClimateControlService started (DQN agent initialized)")

    async def stop(self):
        logger.info("ClimateControlService stopped")

    def pretrain(self, episodes: int = 200):
        sim = TentClimateSimulator()
        self._agent.train_episodes(sim, episodes=episodes, max_steps=48)
        self._trained = True
        logger.info("DQN agent pretrained with %d episodes", episodes)

    def recommend(
        self,
        tent_id: int,
        climate: Dict[str, float],
    ) -> ControlRecommendation:
        if tent_id not in self._simulators:
            self._simulators[tent_id] = TentClimateSimulator(climate)

        sim = self._simulators[tent_id]
        state = sim.reset(climate)
        state_arr = state.to_array()

        if not self._trained:
            self.pretrain(episodes=100)

        try:
            action_idx = self._agent.select_action(state_arr, training=False)
            action = ControlAction.from_index(action_idx)

            next_state, expected_reward = sim.step(action, dt_hours=1.0)

            if expected_reward < 0.05:
                action = self._rule_based_fallback(state)
                sim.reset(climate)
                next_state, expected_reward = sim.step(action, dt_hours=1.0)
                logger.info("DQN reward low (%.3f), using rule-based fallback for tent %d",
                            expected_reward, tent_id)
        except Exception as e:
            logger.warning("DQN recommend failed for tent %d: %s, using rule-based", tent_id, e)
            action = self._rule_based_fallback(state)
            sim.reset(climate)
            next_state, expected_reward = sim.step(action, dt_hours=1.0)

        base_shelf = self._estimate_shelf_life(climate)
        projected_climate = {
            "temperature": next_state.temperature,
            "humidity": next_state.humidity,
            "light": next_state.light,
            "aw": next_state.aw,
        }
        projected_shelf = self._estimate_shelf_life(projected_climate)
        improvement = max(0, projected_shelf - base_shelf)

        return ControlRecommendation(
            tent_id=tent_id,
            action=action,
            expected_reward=round(expected_reward, 4),
            current_state=state,
            projected_state=next_state,
            shelf_life_improvement_days=round(improvement, 1),
            description=action.describe(),
        )

    def _rule_based_fallback(self, state: ClimateState) -> ControlAction:
        vent = 0
        shade = 0
        humid = 0

        if state.temperature > 25:
            vent = 2
        elif state.temperature > 20:
            vent = 1

        if state.light > 500:
            shade = 1

        if state.aw < 0.4:
            humid = 1
        elif state.aw > 0.6:
            vent = min(2, vent + 1)

        return ControlAction(ventilation=vent, shading=shade, humidifier=humid)

    def _estimate_shelf_life(self, climate: Dict[str, float]) -> float:
        temp = climate.get("temperature", 25)
        aw = climate.get("aw", 0.5)
        light = climate.get("light", 300)

        base_days = 365
        temp_factor = max(0.1, 1 - (temp - 20) / 40)
        aw_factor = 1.0 if aw <= 0.5 else max(0.1, 1 - 2 * (aw - 0.5) ** 1.5)
        light_factor = 0.85 if light > 500 else 1.0

        return base_days * temp_factor * aw_factor * light_factor
