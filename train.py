import os
from collections import defaultdict
from functools import partial

import flax.linen as nn
import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
# Tensorboard: Prevent tf from allocating full GPU memory
import tensorflow as tf
import tqdm
from flax.metrics import tensorboard
from flax.training.train_state import TrainState

from sac_jax import SACAgent, StochasticGaussianActor, SingleCritic
# from sac_jax.envs.dmcontrol import make_dmc_env
from sac_jax.data import SequentialReplayBuffer

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
# jax.config.update("jax_disable_jit", True)

@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
  env_config = cfg['env']
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  sac_config = cfg['sac']

  ##############################
  # Logger setup
  ##############################
  output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'tensorboard'))
  writer.hparams(cfg)

  ##############################
  # Environment setup
  ##############################
  def make_env(env_config, seed):
    def make_gym_env(env_id, seed):
      env = gym.make(env_id)
      env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env

    if env_config.backend == "gymnasium":
      return make_gym_env(env_config.env_id, seed)
    elif env_config.backend == "dmc":
      _env = make_dmc_env(env_config.env_id, seed, env_config.dmc.obs_type)
      env = gym.wrappers.RecordEpisodeStatistics(_env)
      return env
    raise ValueError("Environment not supported:", env_config)

  vector_env_cls = gym.vector.AsyncVectorEnv if env_config.asynchronous else gym.vector.SyncVectorEnv
  env = vector_env_cls(
      [
          partial(make_env, env_config, seed)
          for seed in range(cfg.seed, cfg.seed+env_config.num_envs)
      ])
  np.random.seed(cfg.seed)
  rng = jax.random.PRNGKey(cfg.seed)

  ##############################
  # Agent setup
  ##############################
  dtype = jnp.dtype(model_config.dtype)
  rng, actor_key, critic_key1, critic_key2 = jax.random.split(rng, 4)

  ##############################
  # Replay buffer setup
  ##############################
  dummy_obs, _ = env.reset()
  dummy_action = env.action_space.sample()
  dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = \
      env.step(dummy_action)
  replay_buffer = SequentialReplayBuffer(
      capacity=cfg.max_steps//env_config.num_envs,
      num_envs=env_config.num_envs,
      seed=cfg.seed,
      dummy_input=dict(
          observation=dummy_obs,
          action=dummy_action,
          reward=dummy_reward,
          next_observation=dummy_next_obs,
          terminated=dummy_term,
          truncated=dummy_trunc)
  )
  # Actor와 Critic 네트워크 정의
  state_dim = dummy_obs.shape[1]
  action_dim = dummy_action.shape[1]
  actor_model_def = StochasticGaussianActor(action_dim=action_dim, hidden_dim=256)
  critic_model_def = SingleCritic(hidden_dim=256)

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

  agent = SACAgent.create(
      policy_model=actor_state,
      critic_model1=critic_state1,
      critic_model2=critic_state2,
      discount=0.99,
      tau=0.005,
      entropy_coef=0.2,
      batch_size=256,
  )
  global_step = 0

  options = ocp.CheckpointManagerOptions(
      max_to_keep=1, save_interval_steps=cfg['save_interval_steps'])
  checkpoint_path = os.path.join(output_dir, 'checkpoint')
  with ocp.CheckpointManager(
      checkpoint_path, options=options, item_names=(
          'agent', 'global_step', 'buffer_state')
  ) as mngr:
    if mngr.latest_step() is not None:
      print('Checkpoint folder found, restoring from', mngr.latest_step())
      abstract_buffer_state = jax.tree.map(
          ocp.utils.to_shape_dtype_struct, replay_buffer.get_state()
      )
      restored = mngr.restore(mngr.latest_step(),
                              args=ocp.args.Composite(
          agent=ocp.args.StandardRestore(agent),
          global_step=ocp.args.JsonRestore(),
          buffer_state=ocp.args.StandardRestore(abstract_buffer_state),
      )
      )
      agent, global_step = restored.agent, restored.global_step
      replay_buffer.restore(restored.buffer_state)
    else:
      print('No checkpoint folder found, starting from scratch')
      mngr.save(
          global_step,
          args=ocp.args.Composite(
              agent=ocp.args.StandardSave(agent),
              global_step=ocp.args.JsonSave(global_step),
              buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
          ),
      )
      mngr.wait_until_finished()

    ##############################
    # Training loop
    ##############################
    ep_info = {}
    ep_count = np.zeros(env_config.num_envs, dtype=int)
    prev_logged_step = global_step
    observation, _ = env.reset(seed=cfg.seed)

    T = 500
    seed_steps = int(max(5*T, 1000) * env_config.num_envs *
                     env_config.utd_ratio)
    pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)
    for global_step in range(global_step, cfg.max_steps, env_config.num_envs):
      if global_step <= seed_steps:
        action = env.action_space.sample()
      else:
        rng, action_key = jax.random.split(rng)
        action = agent.act(
            observation, key=action_key)
        action = np.array(action)

      next_observation, reward, terminated, truncated, info = env.step(action)

      # Get real final observation and store transition
      real_next_observation = next_observation.copy()
      # for ienv, trunc in enumerate(truncated):
      #   if trunc:
      #     real_next_observation[ienv] = info['final_observation'][ienv]
      replay_buffer.insert(dict(
          observation=observation,
          action=action,
          reward=reward,
          next_observation=real_next_observation,
          terminated=terminated,
          truncated=truncated))
      observation = next_observation

      # Handle terminations/truncations
      done = np.logical_or(terminated, truncated)
      if "episode" in info:
        train_returns = info['episode']['r']
        train_lengths = info['episode']['l']
        for ienv, done_ienv in enumerate(done):
          if done_ienv:
            ep_count[ienv] += 1
            tqdm.tqdm.write(
                f"Episode {ep_count[ienv]} | "
                f"Reward: {info['episode']['r'][ienv]:.2f} | "
                f"Length: {info['episode']['l'][ienv]}"
            )
        writer.scalar(f'episode/train_return',
                      train_returns[train_returns!=0].mean(), global_step)
        writer.scalar(f'episode/train_length',
                      train_lengths[train_lengths!=0].mean(), global_step)
        ep_count[ienv] += 1

      if global_step >= seed_steps:
        if global_step == seed_steps:
          print('Pre-training on seed data...')
          num_updates = seed_steps
        else:
          num_updates = max(1, int(env_config.num_envs * env_config.utd_ratio))

        rng, *update_keys = jax.random.split(rng, num_updates+1)
        log_this_step = global_step >= prev_logged_step + \
            cfg['log_interval_steps']
        if log_this_step:
          all_train_info = defaultdict(list)
          prev_logged_step = global_step

        for iupdate in range(num_updates):
          batch = replay_buffer.sample(agent.batch_size, 1)
          agent, train_info = agent.update(
              observations=batch['observation'],
              actions=batch['action'],
              rewards=batch['reward'],
              next_observations=batch['next_observation'],
              terminals=batch['terminated'],
              key=update_keys[iupdate])

          if log_this_step:
            for k, v in train_info.items():
              all_train_info[k].append(np.array(v))

        if log_this_step:
          for k, v in all_train_info.items():
            writer.scalar(f'train/{k}_mean', np.mean(v), global_step)
            writer.scalar(f'train/{k}_std', np.std(v), global_step)

        mngr.save(
            global_step,
            args=ocp.args.Composite(
                agent=ocp.args.StandardSave(agent),
                global_step=ocp.args.JsonSave(global_step),
                buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
            ),
        )

      pbar.update(env_config.num_envs)
    pbar.close()


if __name__ == '__main__':
  train()
