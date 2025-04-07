from __future__ import annotations
import copy
from functools import partial
from typing import Any, Dict, Tuple, Callable
import flax.linen as nn
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from numpy.typing import ArrayLike
from jaxtyping import PRNGKeyArray
import jax
import jax.numpy as jnp
import optax

# ------------------------------------------------------------------
# Dummy Network Definitions (Actor and Critic)
# ------------------------------------------------------------------
class StochasticGaussianActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    min_log_std: float = -20.0  # log_std의 최솟값
    max_log_std: float = 2.0  # log_std의 최댓값

    @nn.compact
    def __call__(self, x, *, rng=None):
        # 네트워크 전방 계산
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        # log_std 값 클리핑
        log_std = jnp.clip(log_std, self.min_log_std, self.max_log_std)
        std = jnp.exp(log_std)

        # RNG 키 설정: 만약 주어지지 않으면 내부 RNG 사용
        if rng is None:
            rng = self.make_rng("sample")
        noise = jax.random.normal(rng, shape=mean.shape)

        # reparameterization: u = mean + noise * std
        u = mean + noise * std

        # 정규분포의 log 확률 (각 액션 차원에 대해 계산 후 합산)
        pre_tanh_log_prob = -0.5 * (((u - mean) / (std + 1e-6)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
        pre_tanh_log_prob = pre_tanh_log_prob.sum(axis=-1, keepdims=True)

        # tanh squash 적용
        action = jnp.tanh(u)

        # tanh로 인한 확률 보정: log_det_jacobian = sum(log(1 - tanh(u)^2 + eps))
        log_prob = pre_tanh_log_prob - jnp.sum(jnp.log(1 - jnp.square(action) + 1e-6), axis=-1, keepdims=True)

        return action, log_std, log_prob, u


class SingleCritic(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        q = nn.Dense(1)(x)
        return q, None


# ------------------------------------------------------------------
# SACAgent with Twin Critics (Using Flax TrainState)
# ------------------------------------------------------------------
class SACAgent(struct.PyTreeNode):
    # Models: policy, two critics and their target networks
    policy_model: TrainState
    critic_model1: TrainState
    critic_model2: TrainState
    target_critic_model1: TrainState
    target_critic_model2: TrainState

    # SAC hyperparameters (not part of the PyTree)
    discount: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    entropy_coef: float = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls,
               policy_model: TrainState,
               critic_model1: TrainState,
               critic_model2: TrainState,
               discount: float,
               tau: float,
               entropy_coef: float,
               batch_size : int,
               ) -> SACAgent:
        # Copy critic params to target critics
        target_critic_model1 = critic_model1.replace(params=critic_model1.params)
        target_critic_model2 = critic_model2.replace(params=critic_model2.params)
        return cls(
            policy_model=policy_model,
            critic_model1=critic_model1,
            critic_model2=critic_model2,
            target_critic_model1=target_critic_model1,
            target_critic_model2=target_critic_model2,
            discount=discount,
            tau=tau,
            entropy_coef=entropy_coef,
            batch_size = batch_size
        )

    @jax.jit
    def act(self,
            obs: np.ndarray,
            train: bool = True,
            *,
            key: PRNGKeyArray):
        """
        주어진 관측(obs)를 받아 policy 네트워크에서 액션을 샘플링합니다.
        """
        # policy_model.apply_fn는 (params, obs, key) 순으로 받는다고 가정합니다.
        action, _, _, _ = self.policy_model.apply_fn(self.policy_model.params, obs, rngs={'sample': key})
        return action

    @jax.jit
    def update(self,
               observations: jax.Array,
               actions: jax.Array,
               rewards: jax.Array,
               next_observations: jax.Array,
               terminals: jax.Array,
               *,
               key: PRNGKeyArray
               ) -> Tuple[SACAgent, Dict[str, Any]]:
        """
        SAC 업데이트:
          - Twin Critic 업데이트: MSE 손실로 각각 업데이트
          - Policy 업데이트: Q와 entropy를 고려한 손실
          - Target 네트워크: Polyak averaging 업데이트
        """
        # 키 분할
        key, critic_key, policy_key, target_key = jax.random.split(key, 4)

        # 타깃 Q 계산: 다음 상태에서 policy를 통해 액션 샘플링
        next_action, _, log_prob, _ = self.policy_model.apply_fn(self.policy_model.params,
                                                                 next_observations, rngs={'sample': target_key})
        target_Q1, _ = self.target_critic_model1.apply_fn(self.target_critic_model1.params,
                                                          next_observations, next_action, rngs={'sample': target_key})
        target_Q2, _ = self.target_critic_model2.apply_fn(self.target_critic_model2.params,
                                                          next_observations, next_action, rngs={'sample': target_key})
        target_Q = jnp.minimum(target_Q1, target_Q2)
        # SAC 타깃: r + γ (target_Q - α logπ)
        target = rewards[..., None] + self.discount * (1.0 - terminals[..., None]) * (target_Q - self.entropy_coef * log_prob)

        # Twin Critic 손실: 두 Critic의 MSE 손실 합산
        def critic_loss_fn(params1, params2):
            Q1, _ = self.critic_model1.apply_fn(params1, observations, actions)
            Q2, _ = self.critic_model2.apply_fn(params2, observations, actions)
            loss1 = ((Q1 - target) ** 2).mean()
            loss2 = ((Q2 - target) ** 2).mean()
            return loss1 + loss2

        critic_loss, (grads1, grads2) = jax.value_and_grad(critic_loss_fn, argnums=(0, 1))(self.critic_model1.params,
                                                                                           self.critic_model2.params)
        new_critic_model1 = self.critic_model1.apply_gradients(grads=grads1)
        new_critic_model2 = self.critic_model2.apply_gradients(grads=grads2)

        # Policy 손실: (α * log_prob - Q) 를 최소화 (Q는 twin critic의 최소값 사용)
        def policy_loss_fn(policy_params):
            action, _, log_prob, _ = self.policy_model.apply_fn(policy_params, observations, rngs={'sample': policy_key})
            Q1, _ = self.critic_model1.apply_fn(new_critic_model1.params, observations, action)
            Q2, _ = self.critic_model2.apply_fn(new_critic_model2.params, observations, action)
            Q = jnp.minimum(Q1, Q2)
            loss = (self.entropy_coef * log_prob - Q).mean()
            return loss

        policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(self.policy_model.params)
        new_policy_model = self.policy_model.apply_gradients(grads=policy_grads)

        # Target 업데이트: Polyak averaging
        new_target_params1 = optax.incremental_update(new_critic_model1.params,
                                                      self.target_critic_model1.params,
                                                      self.tau)
        new_target_params2 = optax.incremental_update(new_critic_model2.params,
                                                      self.target_critic_model2.params,
                                                      self.tau)
        new_target_critic_model1 = self.target_critic_model1.replace(params=new_target_params1)
        new_target_critic_model2 = self.target_critic_model2.replace(params=new_target_params2)

        new_agent = self.replace(
            policy_model=new_policy_model,
            critic_model1=new_critic_model1,
            critic_model2=new_critic_model2,
            target_critic_model1=new_target_critic_model1,
            target_critic_model2=new_target_critic_model2
        )
        info = {'critic_loss': critic_loss, 'policy_loss': policy_loss}
        return new_agent, info


# ------------------------------------------------------------------
# __main__ 예제: SACAgent 사용 예제
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy 환경 차원 설정
    state_dim = 10
    action_dim = 4

    # 초기 RNG 생성
    rng = jax.random.PRNGKey(0)
    rng, actor_key, critic_key1, critic_key2 = jax.random.split(rng, 4)

    # Actor와 Critic 네트워크 정의
    actor_model_def = Actor(action_dim=action_dim, hidden_dim=256)
    critic_model_def = Critic(hidden_dim=256)

    # TrainState 생성 (Flax의 train_state 사용)
    actor_state = TrainState.create(
        apply_fn=actor_model_def.apply,
        params=actor_model_def.init(actor_key, jnp.ones((1, state_dim))),
        tx=optax.adam(learning_rate=3e-4)
    )
    critic_state1 = TrainState.create(
        apply_fn=critic_model_def.apply,
        params=critic_model_def.init(critic_key1, jnp.ones((1, state_dim)), jnp.ones((1, action_dim))),
        tx=optax.adam(learning_rate=3e-4)
    )
    critic_state2 = TrainState.create(
        apply_fn=critic_model_def.apply,
        params=critic_model_def.init(critic_key2, jnp.ones((1, state_dim)), jnp.ones((1, action_dim))),
        tx=optax.adam(learning_rate=3e-4)
    )

    # SACAgent 생성 (twin critic 포함)
    agent = SACAgent.create(
        policy_model=actor_state,
        critic_model1=critic_state1,
        critic_model2=critic_state2,
        discount=0.99,
        tau=0.005,
        entropy_coef=0.2
    )

    # Dummy 배치 데이터 (업데이트용)
    batch_size = 32
    observations = jnp.ones((batch_size, state_dim))
    actions = jnp.ones((batch_size, action_dim))
    rewards = jnp.ones((batch_size, 1))
    next_observations = jnp.ones((batch_size, state_dim))
    terminals = jnp.zeros((batch_size, 1))

    rng, update_key = jax.random.split(rng)
    agent, info = agent.update(observations, actions, rewards, next_observations, terminals, key=update_key)
    print("SACAgent updated.")
    print("Critic loss:", info['critic_loss'])
    print("Policy loss:", info['policy_loss'])

    # act() 함수 테스트
    test_obs = np.ones((1, state_dim))
    rng, act_key = jax.random.split(rng)
    action = agent.act(test_obs, key=act_key)
    print("Sampled action:", action)

