import time

import numpy as np
import torch as t
import torch.nn as nn
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.distributions.categorical import Categorical
from tqdm.auto import tqdm

from common import ENV_DICT
from gpu_env import get_episode_data_from_infos
from networks import get_actor_and_critic
from objectives import calc_clipped_surrogate_objective, calc_entropy_bonus, calc_value_function_loss
from part1_intro_to_rl.utils import set_global_seeds
from ppo_agent import PPOAgent
from ppo_args import PPOArgs
from ppo_scheduler import make_optimizer
from replay_memory import ReplayMemory
from replay_minibatch import ReplayMinibatch
from rl_utils import LiveVideo, log_grid_video, log_video_html


class PPOTrainer:
    def __init__(self, args: PPOArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.run_name = f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        # Accelerated vectorised env, chosen by mode. All three expose the same gym-style
        # reset()/step() returning GPU tensors, so PPO never leaves the GPU:
        #   classic-control -> GPU CartPole;  atari -> EnvPool (C++ emulators);  mujoco -> Brax (GPU physics).
        self.envs = ENV_DICT[args.mode](args.env_id, args.num_envs, seed=args.seed, device=args.device)

        # Define some basic variables from our environment
        self.num_envs = self.envs.num_envs
        self.action_shape = self.envs.single_action_space.shape
        self.obs_shape = self.envs.single_observation_space.shape

        # Create our replay memory
        self.memory = ReplayMemory(
            self.num_envs,
            self.obs_shape,
            self.action_shape,
            args.batch_size,
            args.minibatch_size,
            args.batches_per_learning_phase,
            args.seed,
        )

        # Create our networks & optimizer
        self.actor, self.critic = get_actor_and_critic(self.envs, mode=args.mode)
        self.actor.to(args.device), self.critic.to(args.device)  # get_actor_and_critic builds them on the default device
        self.optimizer, self.scheduler = make_optimizer(self.actor, self.critic, args.total_phases, args.lr)

        # Create our agent
        self.agent = PPOAgent(self.envs, self.actor, self.critic, self.memory)

    def rollout_phase(self) -> dict | None:
        """
        This function populates the memory with a new set of experiences, using self.agent.play_step
        to step through the environment. It also returns a dict of data which you can include in
        your progress bar postfix.
        """
        data = None
        t0 = time.time()

        # An inner progress bar under train()'s outer one. leave=False removes it when the phase ends,
        # and delay=0.5 only shows it for slow phases (e.g. Atari), so fast CartPole phases don't flicker.
        for step in tqdm(range(self.args.num_steps_per_rollout), desc="rollout", leave=False, delay=0.5):
            # Play a step, returning the infos dict (containing information for each environment)
            infos = self.agent.play_step()

            # Get data from environments, and log it if some environment did actually terminate
            new_data = get_episode_data_from_infos(infos)
            if new_data is not None:
                data = new_data
                if self.args.use_wandb:
                    wandb.log(new_data, step=self.agent.step)

        if self.args.use_wandb:
            wandb.log(
                {"SPS": (self.args.num_steps_per_rollout * self.num_envs) / (time.time() - t0)},
                step=self.agent.step,
            )

        return data

    def learning_phase(self) -> None:
        """
        This function does the following:
            - Generates minibatches from memory
            - Calculates the objective function, and takes an optimization step based on it
            - Clips the gradients (see detail #11)
            - Steps the learning rate scheduler
        """
        minibatches = self.agent.get_minibatches(self.args.gamma, self.args.gae_lambda)
        for minibatch in tqdm(minibatches, desc="learning", leave=False, delay=0.5):  # inner bar, as in rollout_phase
            objective_fn = self.compute_ppo_objective(minibatch)
            objective_fn.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.args.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()

    def compute_ppo_objective(self, minibatch: ReplayMinibatch) -> Float[Tensor, ""]:
        """
        Handles learning phase for a single minibatch. Returns objective function to be maximized.
        """
        logits = self.actor(minibatch.obs)
        dist = Categorical(logits=logits)
        values = self.critic(minibatch.obs).squeeze()

        clipped_surrogate_objective = calc_clipped_surrogate_objective(
            dist, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef
        )
        value_loss = calc_value_function_loss(values, minibatch.returns, self.args.vf_coef)
        entropy_bonus = calc_entropy_bonus(dist, self.args.ent_coef)

        total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus

        with t.inference_mode():
            newlogprob = dist.log_prob(minibatch.actions)
            logratio = newlogprob - minibatch.logprobs
            ratio = logratio.exp()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        # Stash the latest loss components so the training progress bar can display them (and reuse
        # them for the optional wandb log).
        self.last_metrics = dict(
            value_loss=value_loss.item(),
            clip_surr=clipped_surrogate_objective.item(),
            entropy=entropy_bonus.item(),
            approx_kl=approx_kl,
        )
        if self.args.use_wandb:
            wandb.log(
                dict(
                    total_steps=self.agent.step,
                    values=values.mean().item(),
                    lr=self.scheduler.optimizer.param_groups[0]["lr"],
                    clipfrac=np.mean(clipfracs),
                    **self.last_metrics,
                ),
                step=self.agent.step,
            )

        return total_objective_function

    def log_video(self, phase: int, caption: str | None = None) -> None:
        """Render the first 16 envs of the rollout currently sitting in the replay memory as a 4x4
        grid video (drawn with the env's own `draw` method) via the shared `rl_utils.log_grid_video`:
        logged to wandb if `use_wandb`, and displayed inline (into `self.video_slot`, replacing the
        previous video) if `live_viz`. This is what `video_log_freq` does. It reuses the rollout we
        just collected, so it costs no extra env steps. MuJoCo observations can't be drawn, so there
        `BraxEnvs` instead renders the poses it recorded during the last ~300 steps for its first 4
        envs (also free) with the real `mujoco.Renderer`, as a 2x2 grid (~2 s on CPU; see
        `BraxEnvs.render_recent_grid_html`)."""
        if self.args.mode == "mujoco":
            try:
                video = self.envs.render_recent_grid_html()
                log_video_html(video, step=self.agent.step, live=self.args.live_viz, slot=self.video_slot, caption=caption)
            except Exception as e:  # never let visualization break training
                print(f"[video log skipped: {e}]")
            return
        if not callable(getattr(self.envs, "draw", None)):
            return
        try:
            obs = t.stack([o[:16] for o in self.memory.obs], dim=1).cpu()           # (16, T, *obs_shape)
            dones = t.stack([d[:16] for d in self.memory.terminated], dim=1).cpu()  # (16, T)
            if self.args.mode == "pendulum":  # pendulum's draw() wants the applied torque appended
                actions = t.stack([a[:16] for a in self.memory.actions], dim=1).cpu()
                obs = t.cat([obs, actions.reshape(*obs.shape[:2], -1)], dim=-1)
            cell_w, cell_h = (84, 84) if self.args.mode == "atari" else (160, 120)
        except Exception as e:  # never let visualization break training
            print(f"[video log skipped: {e}]")
            return
        log_grid_video(
            obs, self.envs.draw, dones=dones, step=self.agent.step, live=self.args.live_viz,
            slot=self.video_slot, caption=caption, cell_w=cell_w, cell_h=cell_h,
        )

    def train(self) -> None:
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
            )
            wandb.watch([self.actor, self.critic], log="all", log_freq=50)

        # The bar counts env steps (one phase = batch_size of them), so tqdm shows the throughput in SI units.
        pbar = tqdm(total=self.args.total_timesteps, unit=" env steps", unit_scale=True, desc=f"training {self.args.mode}")
        last_logged_time = time.time()  # so we don't update the progress bar too much
        data = {}
        self.video_slot = LiveVideo(overwrite=self.args.overwrite_video)  # one display slot per run (see rl_utils.LiveVideo)
        want_video = self.args.video_log_freq and (self.args.use_wandb or self.args.live_viz)

        for phase in range(self.args.total_phases):
            new_data = self.rollout_phase()
            if new_data is not None:
                data = new_data
            # Periodically render the rollout we just collected as a 4x4 grid video (must happen here,
            # before the learning phase consumes & resets the memory) - to wandb if use_wandb, inline if live_viz.
            if want_video and (phase % self.args.video_log_freq == 0):
                caption = f"phase {phase}/{self.args.total_phases}"
                if "episode_reward" in data:
                    caption += f" | last episode: reward {data['episode_reward']:.1f}, length {data['episode_length']:.0f}"
                self.log_video(phase, caption=caption)
            self.learning_phase()
            pbar.update(self.args.batch_size)
            # Show episode stats (when available) alongside the latest loss components / KL.
            if time.time() - last_logged_time > 0.5:
                last_logged_time = time.time()
                pbar.set_postfix(phase=phase, **data, **getattr(self, "last_metrics", {}))
        pbar.close()

        self.envs.close()
        if self.args.use_wandb:
            # Remove the watch() forward hooks BEFORE finishing the run: they log to `wandb.run`,
            # which becomes None after finish(), so any later forward pass through the networks
            # (e.g. rendering a video of the trained agent) would crash with
            # "'NoneType' object has no attribute '_log'".
            wandb.unwatch((self.actor, self.critic))
            wandb.finish()
