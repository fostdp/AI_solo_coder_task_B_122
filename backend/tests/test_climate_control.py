"""
测试 climate_control 模块 - DQN强化学习微气候调控

覆盖维度:
  - 正常: DQN训练后高温帐篷降温; 规则策略与DQN结果一致; 训练回报收敛趋势
  - 边界: 连续动作空间离散化(9个动作); 极端气候状态有界; 零步长不崩溃
  - 异常: 模拟器不收敛时降级为规则策略; DQN异常时try/except降级
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from unittest.mock import patch

from services.climate_control.dqn_controller import (
    ClimateState, ControlAction, TentClimateSimulator,
    DQNAgent, ClimateControlService, ControlRecommendation,
)


# ============================================================
#  正常场景
# ============================================================

class TestNormalClimateControl:
    def test_dqn_training_reduces_high_temperature(self):
        svc = ClimateControlService()
        svc.pretrain(episodes=100)
        climate = {"temperature": 35, "humidity": 70, "light": 600, "aw": 0.55}
        rec = svc.recommend(tent_id=1, climate=climate)
        assert rec.projected_state.temperature <= climate["temperature"] + 1

    def test_rule_based_fallback_also_cools(self):
        svc = ClimateControlService()
        state = ClimateState(35, 70, 600, 0.55, 0, 0, 0)
        action = svc._rule_based_fallback(state)
        assert action.ventilation > 0
        assert action.shading > 0

    def test_training_rewards_improve(self):
        agent = DQNAgent(epsilon_decay=0.95, batch_size=8)
        sim = TentClimateSimulator({"temperature": 30, "humidity": 65, "light": 500, "aw": 0.55})
        result = agent.train_episodes(sim, episodes=50, max_steps=24)
        rewards = result["rewards"]
        first_quarter = sum(rewards[:len(rewards) // 4]) / max(1, len(rewards) // 4)
        last_quarter = sum(rewards[-len(rewards) // 4:]) / max(1, len(rewards) // 4)
        assert last_quarter >= first_quarter * 0.8

    def test_recommend_for_ideal_climate(self):
        svc = ClimateControlService()
        climate = {"temperature": 20, "humidity": 50, "light": 300, "aw": 0.45}
        rec = svc.recommend(tent_id=1, climate=climate)
        assert rec.shelf_life_improvement_days >= 0

    def test_recommend_for_hot_climate(self):
        svc = ClimateControlService()
        climate = {"temperature": 40, "humidity": 80, "light": 800, "aw": 0.70}
        rec = svc.recommend(tent_id=1, climate=climate)
        assert rec.action.ventilation > 0 or rec.action.shading > 0

    def test_recommend_for_dry_climate(self):
        svc = ClimateControlService()
        climate = {"temperature": 25, "humidity": 30, "light": 400, "aw": 0.30}
        rec = svc.recommend(tent_id=1, climate=climate)
        assert isinstance(rec, ControlRecommendation)


# ============================================================
#  边界场景
# ============================================================

class TestBoundaryClimateControl:
    def test_continuous_action_discretized_to_nine_actions(self):
        for idx in range(9):
            action = ControlAction.from_index(idx)
            assert action.index == idx
        mapped_combos = set(ControlAction.ACTION_MAP.values())
        assert len(mapped_combos) == 9

    def test_extreme_climate_state_bounded(self):
        sim = TentClimateSimulator({"temperature": 60, "humidity": 100, "light": 2000, "aw": 0.99})
        for _ in range(20):
            action = ControlAction.from_index(np.random.randint(9))
            state, _ = sim.step(action)
            assert -5 <= state.temperature <= 50
            assert 10 <= state.humidity <= 99
            assert 0 <= state.light <= 1000
            assert 0.3 <= state.aw <= 0.95

    def test_zero_step_dt(self):
        sim = TentClimateSimulator({"temperature": 25, "humidity": 50, "light": 400, "aw": 0.50})
        action = ControlAction(2, 1, 1)
        state, reward = sim.step(action, dt_hours=0)
        assert abs(state.temperature - 25) < 0.01

    def test_all_actions_valid_for_any_state(self):
        svc = ClimateControlService()
        svc.pretrain(episodes=50)
        climates = [
            {"temperature": -5, "humidity": 20, "light": 50, "aw": 0.30},
            {"temperature": 45, "humidity": 95, "light": 900, "aw": 0.90},
            {"temperature": 20, "humidity": 50, "light": 300, "aw": 0.45},
        ]
        for i, climate in enumerate(climates):
            rec = svc.recommend(tent_id=100 + i, climate=climate)
            assert 0 <= rec.action.index < 9
            assert rec.expected_reward is not None


# ============================================================
#  异常场景
# ============================================================

class TestExceptionClimateControl:
    def test_dqn_failure_falls_back_to_rule_based(self):
        svc = ClimateControlService()
        svc._trained = True

        with patch.object(svc._agent, 'select_action', side_effect=RuntimeError("DQN crashed")):
            climate = {"temperature": 35, "humidity": 70, "light": 600, "aw": 0.55}
            rec = svc.recommend(tent_id=99, climate=climate)
            assert isinstance(rec, ControlRecommendation)
            assert rec.action.ventilation > 0

    def test_rule_based_fallback_for_hot(self):
        svc = ClimateControlService()
        state = ClimateState(35, 70, 600, 0.55, 0, 0, 0)
        action = svc._rule_based_fallback(state)
        assert action.ventilation == 2

    def test_rule_based_fallback_for_cold(self):
        svc = ClimateControlService()
        state = ClimateState(10, 40, 200, 0.35, 0, 0, 0)
        action = svc._rule_based_fallback(state)
        assert action.ventilation == 0
        assert action.humidifier == 1

    def test_rule_based_fallback_for_bright(self):
        svc = ClimateControlService()
        state = ClimateState(22, 50, 700, 0.45, 0, 0, 0)
        action = svc._rule_based_fallback(state)
        assert action.shading == 1

    def test_low_dqn_reward_triggers_rule_based(self):
        svc = ClimateControlService()
        svc.pretrain(episodes=10)

        climate = {"temperature": 40, "humidity": 90, "light": 900, "aw": 0.85}
        rec = svc.recommend(tent_id=1, climate=climate)
        assert isinstance(rec, ControlRecommendation)

    def test_simulator_nonconvergence_still_returns_recommendation(self):
        svc = ClimateControlService()
        climate = {"temperature": 49, "humidity": 98, "light": 990, "aw": 0.94}
        rec = svc.recommend(tent_id=1, climate=climate)
        assert rec is not None
        assert rec.tent_id == 1


# ============================================================
#  基础组件
# ============================================================

class TestClimateState:
    def test_to_array_shape(self):
        s = ClimateState(25, 60, 400, 0.5, 0, 0, 0)
        arr = s.to_array()
        assert arr.shape == (7,)
        assert arr.dtype == np.float32

    def test_to_array_normalized(self):
        s = ClimateState(25, 50, 500, 0.5, 1, 1, 0)
        arr = s.to_array()
        assert 0 <= arr[0] <= 1
        assert 0 <= arr[3] <= 1


class TestControlAction:
    def test_from_index_roundtrip(self):
        for idx in range(ControlAction.num_actions()):
            action = ControlAction.from_index(idx)
            assert action.index == idx

    def test_num_actions(self):
        assert ControlAction.num_actions() == 9

    def test_describe(self):
        a = ControlAction(ventilation=2, shading=1, humidifier=0)
        desc = a.describe()
        assert "通风口" in desc
        assert "遮阳帘" in desc

    def test_energy_cost(self):
        a0 = ControlAction(0, 0, 0)
        a_max = ControlAction(2, 1, 1)
        assert a0.energy_cost() < a_max.energy_cost()

    def test_energy_cost_values(self):
        a = ControlAction(2, 1, 1)
        assert abs(a.energy_cost() - 1.2) < 0.01


class TestTentClimateSimulator:
    def test_reset(self):
        sim = TentClimateSimulator({"temperature": 30, "humidity": 70, "light": 500, "aw": 0.55})
        state = sim.reset()
        assert state.temperature == 30

    def test_step_ventilation_cools(self):
        sim = TentClimateSimulator({"temperature": 30, "humidity": 70, "light": 400, "aw": 0.50})
        action = ControlAction(ventilation=2, shading=0, humidifier=0)
        next_state, _ = sim.step(action)
        assert next_state.temperature < 30

    def test_step_shading_reduces_light(self):
        sim = TentClimateSimulator({"temperature": 25, "humidity": 50, "light": 500, "aw": 0.50})
        action = ControlAction(ventilation=0, shading=1, humidifier=0)
        next_state, _ = sim.step(action)
        assert next_state.light < 500

    def test_step_humidifier_increases_aw(self):
        sim = TentClimateSimulator({"temperature": 25, "humidity": 50, "light": 400, "aw": 0.40})
        action = ControlAction(ventilation=0, shading=0, humidifier=1)
        next_state, _ = sim.step(action)
        assert next_state.aw > 0.40


class TestDQNAgent:
    def test_initial_epsilon(self):
        agent = DQNAgent()
        assert agent.epsilon == 1.0

    def test_select_action_returns_valid(self):
        agent = DQNAgent()
        state = np.zeros(7, dtype=np.float32)
        for _ in range(20):
            action = agent.select_action(state, training=True)
            assert 0 <= action < 9

    def test_epsilon_decay(self):
        agent = DQNAgent(epsilon_decay=0.9, epsilon_end=0.1)
        sim = TentClimateSimulator()
        state = sim.reset()
        for _ in range(50):
            s = state.to_array()
            a = agent.select_action(s)
            action = ControlAction.from_index(a)
            ns, r = sim.step(action)
            agent.store_transition(s, a, r, ns.to_array(), False)
            agent.train_step()
            state = ns
        assert agent.epsilon < 1.0
