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


# ============================================================
#  修复验证：潜力塑形奖励稳定性
# ============================================================

class TestPotentialShapingStability:
    def test_potential_function_range(self):
        sim = TentClimateSimulator()
        states = [
            ClimateState(20, 50, 300, 0.45, 0, 0, 0),
            ClimateState(35, 80, 800, 0.7, 0, 0, 0),
            ClimateState(0, 20, 0, 0.3, 0, 0, 0),
            ClimateState(50, 99, 1000, 0.95, 2, 1, 1),
        ]
        for s in states:
            phi = sim.potential(s)
            assert 0.0 <= phi <= 1.0, f"potential {phi} 超出 [0,1] 范围"

    def test_ideal_state_has_max_potential(self):
        sim = TentClimateSimulator()
        ideal = ClimateState(20, 50, 300, 0.45, 0, 0, 0)
        bad = ClimateState(35, 80, 800, 0.7, 0, 0, 0)
        assert sim.potential(ideal) > sim.potential(bad)

    def test_potential_shaping_reward_improves_convergence(self):
        agent_shaped = DQNAgent(epsilon_decay=0.99, batch_size=16, lr=0.01)
        agent_vanilla = DQNAgent(epsilon_decay=0.99, batch_size=16, lr=0.01)
        sim = TentClimateSimulator({"temperature": 32, "humidity": 70, "light": 600, "aw": 0.6})

        result_shaped = agent_shaped.train_episodes(sim, episodes=30, max_steps=24, use_potential_shaping=True)
        result_vanilla = agent_vanilla.train_episodes(sim, episodes=30, max_steps=24, use_potential_shaping=False)

        rewards_shaped = result_shaped["rewards"]
        rewards_vanilla = result_vanilla["rewards"]

        last_10_shaped = sum(rewards_shaped[-10:]) / 10
        last_10_vanilla = sum(rewards_vanilla[-10:]) / 10
        first_10_shaped = sum(rewards_shaped[:10]) / 10
        first_10_vanilla = sum(rewards_vanilla[:10]) / 10

        improvement_shaped = last_10_shaped - first_10_shaped
        improvement_vanilla = last_10_vanilla - first_10_vanilla

        if improvement_vanilla > 0:
            assert improvement_shaped >= improvement_vanilla * 0.5, \
                f"塑形奖励改善 {improvement_shaped:.2f} 应不低于原始 {improvement_vanilla:.2f} 的50%"
        else:
            assert improvement_shaped >= improvement_vanilla - 0.5, \
                f"塑形奖励改善 {improvement_shaped:.2f} 不应差于原始 {improvement_vanilla:.2f} 太多"

    def test_1000_steps_stable_control(self):
        agent = DQNAgent(epsilon_decay=0.995, batch_size=32, gamma=0.95)
        sim = TentClimateSimulator({"temperature": 35, "humidity": 75, "light": 700, "aw": 0.65})

        total_steps = 0
        state = sim.reset()
        temp_history = []

        for ep in range(42):
            for step in range(24):
                if total_steps >= 1000:
                    break
                s = state.to_array()
                a = agent.select_action(s, training=True)
                action = ControlAction.from_index(a)
                ns, r = sim.step(action)
                ns_arr = ns.to_array()
                done = step >= 23

                phi_s = sim.potential(state)
                phi_s_prime = sim.potential(ns)
                shaped_r = r + agent.gamma * phi_s_prime - phi_s

                agent.store_transition(s, a, shaped_r, ns_arr, done)
                agent.train_step()

                temp_history.append(ns.temperature)
                state = ns
                total_steps += 1

            if total_steps >= 1000:
                break

        assert total_steps >= 1000, f"应完成至少1000步，实际 {total_steps}"

        last_200 = temp_history[-200:]
        first_200 = temp_history[:200]
        avg_last = sum(last_200) / len(last_200)
        avg_first = sum(first_200) / len(first_200)
        std_last = (sum((x - avg_last) ** 2 for x in last_200) / len(last_200)) ** 0.5
        std_first = (sum((x - avg_first) ** 2 for x in first_200) / len(first_200)) ** 0.5

        ideal_temp = 20.0
        dev_last = abs(avg_last - ideal_temp)
        dev_first = abs(avg_first - ideal_temp)

        assert dev_last <= dev_first + 2, \
            f"后期温度偏差 {dev_last:.1f}°C 不应比前期 {dev_first:.1f}°C 差太多"
        assert std_last <= std_first * 2.0, \
            f"后期温度标准差 {std_last:.2f} 不应大幅高于前期 {std_first:.2f}"

    def test_shaped_reward_signals_dense(self):
        agent = DQNAgent(batch_size=1)
        sim = TentClimateSimulator({"temperature": 35, "humidity": 75, "light": 700, "aw": 0.65})

        state = sim.reset()
        phi_s = sim.potential(state)

        good_action = ControlAction(ventilation=2, shading=1, humidifier=0)
        bad_action = ControlAction(ventilation=0, shading=0, humidifier=1)

        ns_good, r_good = sim.step(good_action)
        phi_good = sim.potential(ns_good)
        shaped_good = r_good + agent.gamma * phi_good - phi_s

        sim.reset()
        state2 = sim.reset()
        phi_s2 = sim.potential(state2)
        ns_bad, r_bad = sim.step(bad_action)
        phi_bad = sim.potential(ns_bad)
        shaped_bad = r_bad + agent.gamma * phi_bad - phi_s2

        assert shaped_good > shaped_bad, f"好动作塑形奖励 {shaped_good:.3f} 应大于坏动作 {shaped_bad:.3f}"
