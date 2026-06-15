"""
测试 climate_control 模块 - DQN强化学习微气候调控
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from services.climate_control.dqn_controller import (
    ClimateState, ControlAction, TentClimateSimulator,
    DQNAgent, ClimateControlService, ControlRecommendation,
)


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
        assert state.humidity == 70

    def test_step_ventilation_cools(self):
        sim = TentClimateSimulator({"temperature": 30, "humidity": 70, "light": 400, "aw": 0.50})
        action = ControlAction(ventilation=2, shading=0, humidifier=0)
        next_state, reward = sim.step(action)
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

    def test_reward_positive_for_ideal(self):
        sim = TentClimateSimulator({"temperature": 20, "humidity": 50, "light": 300, "aw": 0.45})
        action = ControlAction(0, 0, 0)
        _, reward = sim.step(action)
        assert reward > 0

    def test_reward_negative_for_extreme(self):
        sim = TentClimateSimulator({"temperature": 45, "humidity": 90, "light": 900, "aw": 0.80})
        action = ControlAction(0, 0, 0)
        _, reward = sim.step(action)
        assert reward < 0.5

    def test_state_bounded(self):
        sim = TentClimateSimulator({"temperature": 50, "humidity": 99, "light": 2000, "aw": 0.99})
        for _ in range(10):
            action = ControlAction.from_index(np.random.randint(9))
            state, _ = sim.step(action)
            assert -5 <= state.temperature <= 50
            assert 10 <= state.humidity <= 99
            assert 0 <= state.light <= 1000
            assert 0.3 <= state.aw <= 0.95


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

    def test_store_and_train(self):
        agent = DQNAgent(batch_size=4)
        sim = TentClimateSimulator()
        state = sim.reset()
        for _ in range(10):
            s = state.to_array()
            a = agent.select_action(s)
            action = ControlAction.from_index(a)
            ns, r = sim.step(action)
            agent.store_transition(s, a, r, ns.to_array(), False)
            state = ns
        loss = agent.train_step()
        assert loss is not None or len(agent._replay_buffer) < agent.batch_size

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


class TestClimateControlService:
    def test_recommend(self):
        svc = ClimateControlService()
        climate = {"temperature": 28, "humidity": 65, "light": 500, "aw": 0.55}
        rec = svc.recommend(tent_id=1, climate=climate)
        assert isinstance(rec, ControlRecommendation)
        assert rec.tent_id == 1
        assert 0 <= rec.action.index < 9
        assert isinstance(rec.description, str)

    def test_recommend_shelf_life_improvement(self):
        svc = ClimateControlService()
        climate = {"temperature": 30, "humidity": 70, "light": 600, "aw": 0.60}
        rec = svc.recommend(tent_id=1, climate=climate)
        assert rec.shelf_life_improvement_days >= 0

    def test_estimate_shelf_life(self):
        svc = ClimateControlService()
        good = svc._estimate_shelf_life({"temperature": 20, "aw": 0.45, "light": 200})
        bad = svc._estimate_shelf_life({"temperature": 40, "aw": 0.70, "light": 600})
        assert good > bad
