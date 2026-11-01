import time

import torch as t
import wandb
from tqdm.auto import tqdm

from cartpole import CartPole
from common import set_global_seeds
from dqn_agent import DQNAgent
from dqn_args import DQNArgs
from env_registry import make_envs
from q_network import QNetwork
from replay_buffer import ReplayBuffer
from video import LiveVideo, log_greedy_rollout_video


def get_episode_data_from_infos(infos: dict) -> dict[str, int | float] | None:
    """
    Helper function: returns a dict of episode data (length, reward and duration, averaged over the
    environments that finished an episode on this step), if at least one
    terminated.
    """
    for final_info in infos.get("final_info", []):
        if final_info is not None and "episode" in final_info:
            return {
                "episode_length": final_info["episode"]["l"].item(),
                "episode_reward": final_info["episode"]["r"].item(),
                "episode_duration": final_info["episode"]["t"].item(),
            }


class DQNTrainer:
    def __init__(self, args: DQNArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.rng = t.Generator(device=args.device).manual_seed(args.seed)
        self.run_name = f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"

        self.envs = make_envs(args.env_id, args.num_envs, args.seed, args.device)

        # Define some basic variables from our environment (note, we assume a single discrete action space)
        num_envs = self.envs.num_envs
        action_shape = self.envs.single_action_space.shape
        num_actions = self.envs.single_action_space.n
        obs_shape = self.envs.single_observation_space.shape
        assert action_shape == ()

        # Create our replay buffer (on the same device as the environment)
        self.buffer = ReplayBuffer(num_envs, obs_shape, action_shape, args.buffer_size, args.seed, args.device)

        # Create our networks & optimizer (target network should be initialized with a copy of the Q-network's weights)
        self.q_network = QNetwork(obs_shape, num_actions).to(args.device)
        self.target_network = QNetwork(obs_shape, num_actions).to(args.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = t.optim.AdamW(self.q_network.parameters(), lr=args.learning_rate)

        # Create our agent
        self.agent = DQNAgent(
            self.envs,
            self.buffer,
            self.q_network,
            args.start_e,
            args.end_e,
            args.exploration_fraction,
            args.total_timesteps,
            self.rng,
        )

    def add_to_replay_buffer(self, n: int, verbose: bool = False):
        """
        Calls `agent.play_step` n times (n * num_envs environment steps), adding to the replay buffer.
        Should return a dict of data from the last terminated episode, if any (the training loop takes
        care of logging it).

        Optional argument `verbose`: if True, we can use a progress bar (useful to check how long
        the initial buffer filling is taking).
        """
        data = None

        for step in tqdm(range(n), disable=not verbose, desc="Adding to replay buffer"):
            infos = self.agent.play_step()

            # Get data from environments, if some environment did actually terminate
            new_data = get_episode_data_from_infos(infos)
            if new_data is not None:
                data = new_data  # makes sure we return a non-empty dict, if some episode terminates

        return data

    def prepopulate_replay_buffer(self):
        """
        Called to fill the replay buffer before training starts.
        """
        n_steps_to_fill_buffer = self.args.buffer_size // self.args.num_envs
        self.add_to_replay_buffer(n_steps_to_fill_buffer, verbose=True)

    def training_step(self, step: int) -> None:
        """
        Samples once from the replay buffer, and takes a single optimization step.

        Args:
            step (int): The number of optimization steps taken so far (used for logging, and for deciding
            when to update the target network)
        """
        data = self.buffer.sample(self.args.batch_size)  # o_t, a_t, r_{t+1}, d_{t+1}, o_{t+1}

        with t.inference_mode():
            target_max = self.target_network(data.next_obs).max(-1).values
        predicted_q_vals = self.q_network(data.obs).gather(-1, data.actions.unsqueeze(-1)).squeeze(-1)
        # alternatively with eindex
        # predicted_q_vals = eindex(self.q_network(data.obs), data.actions, "b [b] -> b")
        # alternatively with indexing
        # self.q_network(data.obs)[range(len(data.actions)), data.actions]

        td_error = data.rewards + self.args.gamma * target_max * (1 - data.terminated.float()) - predicted_q_vals
        loss = td_error.pow(2).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if step % self.args.trains_per_target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Log every 10th step
        if self.args.use_wandb and step % 10 == 0:
            wandb.log(
                {
                    "td_loss": loss,
                    "q_values": predicted_q_vals.mean().item(),
                    "epsilon": self.agent.epsilon,
                },
                step=self.agent.step,
            )

    def train(self) -> None:
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
            )

        self.prepopulate_replay_buffer()

        pbar = tqdm(range(self.args.total_training_steps))
        last_logged_time, last_logged_step = time.time(), self.agent.step  # so we don't log too often

        greedy = lambda obs: self.q_network(obs).argmax(-1)
        fresh_envs = lambda seed: CartPole(num_envs=16, seed=seed, device=self.args.device)
        can_render = self.args.env_id == "CartPole-gpu"
        live_videos = can_render and self.args.steps_per_live_video is not None
        wandb_videos = can_render and self.args.use_wandb and self.args.video_log_freq is not None
        video_slot = LiveVideo(overwrite=self.args.overwrite_video)  # one display slot per run (see video.LiveVideo)

        for step in pbar:
            data = self.add_to_replay_buffer(self.args.steps_per_update)
            if data is not None:
                if time.time() - last_logged_time > 0.5:
                    now = time.time()
                    if self.args.use_wandb:
                        sps = (self.agent.step - last_logged_step) / (now - last_logged_time)
                        wandb.log({**data, "SPS": sps}, step=self.agent.step)
                    last_logged_time, last_logged_step = now, self.agent.step
                    pbar.set_postfix(**data)

            self.training_step(step)

            live = live_videos and step % self.args.steps_per_live_video == 0
            to_wandb = wandb_videos and step % self.args.video_log_freq == 0
            if live or to_wandb:
                log_greedy_rollout_video(fresh_envs(step), greedy, self.envs.draw, step=self.agent.step, live=live, slot=video_slot)

        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()
