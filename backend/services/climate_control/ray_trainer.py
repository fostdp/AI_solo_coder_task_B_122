"""
DQN 分布式训练器（基于 Ray）
支持分布式经验回放、并行环境采样，加速训练收敛

注意：Ray 为可选依赖，不可用时自动降级为单进程训练
"""
import time
import logging
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


if RAY_AVAILABLE and not ray.is_initialized():
    try:
        ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False)
        logger.info("Ray initialized for DQN distributed training")
    except Exception as e:
        logger.warning(f"Ray init failed: {e}, using single-process training")
        RAY_AVAILABLE = False


class RayDQNTrainer:
    """Ray 分布式 DQN 训练器"""

    def __init__(self, agent, simulator_factory, num_workers: int = 2):
        self.agent = agent
        self.simulator_factory = simulator_factory
        self.num_workers = num_workers if RAY_AVAILABLE else 1
        self._ray_actors = []

        if RAY_AVAILABLE and num_workers > 1:
            self._init_ray_workers()

    def _init_ray_workers(self):
        """初始化 Ray worker actors"""
        if not RAY_AVAILABLE:
            return

        @ray.remote
        class RolloutWorker:
            def __init__(self, simulator_factory):
                self.sim = simulator_factory()

            def collect_episode(self, agent_weights, epsilon, max_steps):
                import numpy as np
                state = self.sim.reset()
                transitions = []
                total_reward = 0.0

                for step in range(max_steps):
                    state_arr = state.to_array()
                    if np.random.random() < epsilon:
                        from services.climate_control.dqn_controller import ControlAction
                        import random
                        action_idx = random.randint(0, len(ControlAction.ACTION_MAP) - 1)
                    else:
                        state_disc = self._discretize(state_arr, agent_weights["state_bins"])
                        action_idx = int(np.argmax(agent_weights["q_table"][state_disc]))

                    from services.climate_control.dqn_controller import ControlAction
                    action = ControlAction.from_index(action_idx)
                    next_state, reward = self.sim.step(action)

                    phi_s = self.sim.potential(state) if hasattr(self.sim, 'potential') else 0
                    phi_s_prime = self.sim.potential(next_state) if hasattr(self.sim, 'potential') else 0
                    shaped_reward = reward + agent_weights["gamma"] * phi_s_prime - phi_s

                    transitions.append((
                        state_arr.tolist(), action_idx, shaped_reward,
                        next_state.to_array().tolist(), step >= max_steps - 1
                    ))
                    total_reward += reward
                    state = next_state

                return transitions, total_reward

            def _discretize(self, state_arr, bins):
                idx = 0
                for i, val in enumerate(state_arr):
                    bin_list = bins[i]
                    b = min(len(bin_list) - 1, max(0, int(val)))
                    idx = idx * len(bin_list) + b
                return idx

        self._RolloutWorker = RolloutWorker
        self._ray_actors = [
            RolloutWorker.remote(self.simulator_factory)
            for _ in range(self.num_workers)
        ]

    def train(
        self,
        total_episodes: int = 500,
        max_steps: int = 48,
        use_potential_shaping: bool = True,
    ) -> Dict[str, List[float]]:
        """
        训练 DQN 智能体
        - Ray 可用时：并行采样 + 集中更新
        - Ray 不可用时：降级为单进程训练
        """
        if not RAY_AVAILABLE or self.num_workers <= 1:
            sim = self.simulator_factory()
            return self.agent.train_episodes(
                sim, episodes=total_episodes, max_steps=max_steps,
                use_potential_shaping=use_potential_shaping,
            )

        return self._train_distributed(
            total_episodes, max_steps, use_potential_shaping
        )

    def _train_distributed(
        self, total_episodes: int, max_steps: int, use_potential_shaping: bool,
    ) -> Dict[str, List[float]]:
        """分布式训练循环"""
        rewards_history = []
        losses = []
        update_freq = self.num_workers * 5

        for ep_batch in range(total_episodes // update_freq):
            epsilon = max(
                self.agent.epsilon_end,
                self.agent.epsilon * (self.agent.epsilon_decay ** (ep_batch * update_freq))
            )
            self.agent.epsilon = epsilon

            agent_weights = {
                "q_table": self.agent.q_table.tolist(),
                "state_bins": [b.tolist() for b in self.agent.state_bins],
                "gamma": self.agent.gamma,
            }

            futures = [
                worker.collect_episode.remote(agent_weights, epsilon, max_steps)
                for worker in self._ray_actors
            ]
            results = ray.get(futures)

            for transitions, reward in results:
                for s, a, r, ns, d in transitions:
                    import numpy as np
                    self.agent.store_transition(
                        np.array(s, dtype=np.float32), a, r,
                        np.array(ns, dtype=np.float32), d
                    )
                rewards_history.append(reward)

            for _ in range(update_freq):
                loss = self.agent.train_step()
                if loss is not None:
                    losses.append(loss)

            if ep_batch % 10 == 0:
                avg_reward = sum(rewards_history[-10:]) / min(10, len(rewards_history))
                logger.info(f"Batch {ep_batch}: avg_reward={avg_reward:.3f}, epsilon={epsilon:.3f}")

        return {"rewards": rewards_history, "losses": losses}

    def shutdown(self):
        """关闭 Ray actors"""
        if self._ray_actors:
            for actor in self._ray_actors:
                ray.kill(actor)
            self._ray_actors = []


def is_ray_available() -> bool:
    """检查 Ray 是否可用"""
    return RAY_AVAILABLE
