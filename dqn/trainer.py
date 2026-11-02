import torch as t
import torch.nn.functional as F
from tqdm.auto import tqdm

from common import set_global_seeds
from dqn.agent import DQNAgent
from dqn.args import DQNArgs
from dqn.networks import ConvQNetwork, QNetwork
from dqn.replay_buffer import ReplayBuffer, ReplayBufferSamples
from envs.gpu_env import GPUEnv


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
    """
    Trains a DQN agent on `envs`, any batched environment with the GPUEnv interface (envs/gpu_env.py). The caller
    builds and seeds the environment; the trainer reads everything it needs (number of envs, observation and action
    spaces) from it.
    """

    def __init__(self, args: DQNArgs, envs: GPUEnv):
        set_global_seeds(args.seed)
        self.args = args
        self.envs = envs
        self.rng = t.Generator(device=args.device).manual_seed(args.seed)

        # Define some basic variables from our environment (note, we assume a single discrete action space)
        num_envs = self.envs.num_envs
        action_shape = self.envs.single_action_space.shape
        num_actions = self.envs.single_action_space.n
        obs_shape = self.envs.single_observation_space.shape
        assert action_shape == ()

        # Every call to `agent.play_step` advances all `num_envs` environments by one step; we take one optimization
        # step per `steps_per_update` such calls, so this is the total number of optimization steps
        assert args.total_timesteps - args.learning_starts >= args.steps_per_update * num_envs
        self.total_training_steps = (args.total_timesteps - args.learning_starts) // (args.steps_per_update * num_envs)

        # Create our replay buffer (on the same device as the environment)
        self.buffer = ReplayBuffer(num_envs, obs_shape, action_shape, args.buffer_size, args.seed, args.device,
                                   death_share=args.buffer_death_share)

        # Create our networks & optimizer (target network should be initialized with a copy of the Q-network's weights)
        # Image observations (channels, height, width) get the gridworld CNN; flat vectors (probe envs) get the MLP
        network_cls = ConvQNetwork if len(obs_shape) == 3 else QNetwork
        self.q_network = network_cls(obs_shape, num_actions).to(args.device)
        self.target_network = network_cls(obs_shape, num_actions).to(args.device)
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
        warm_up_play_steps = self.args.learning_starts // self.envs.num_envs
        self.add_to_replay_buffer(warm_up_play_steps, verbose=True)

    def td_loss(self, data: ReplayBufferSamples) -> t.Tensor:
        """
        The TD loss on a batch of transitions, with Double DQN's target (van Hasselt et al. 2016): the Q-network chooses
        the best next action and the target network values it. Plain DQN does both with the target network, and
        choosing and valuing with the same noisy estimates favours actions that happen to be overestimated.
        """
        with t.inference_mode():
            best_next = self.q_network(data.next_obs).argmax(-1, keepdim=True)
            next_value = self.target_network(data.next_obs).gather(-1, best_next).squeeze(-1)
        predicted_q_vals = self.q_network(data.obs).gather(-1, data.actions.unsqueeze(-1)).squeeze(-1)
        td_target = data.rewards + self.args.gamma * next_value * (1 - data.terminated.float())
        # Huber loss: squared for small TD errors, linear for large ones, so one rare large error (e.g. a death) can't
        # cause a huge update. This and gradient clipping are standard in DQN (Mnih et al. 2015's error clipping).
        return F.smooth_l1_loss(predicted_q_vals, td_target)

    def training_step(self, step: int) -> None:
        """
        Samples once from the replay buffer, and takes a single optimization step.

        Args:
            step (int): The number of optimization steps taken so far (used for logging, and for deciding
            when to update the target network)
        """
        data = self.buffer.sample(self.args.batch_size)  # o_t, a_t, r_{t+1}, d_{t+1}, o_{t+1}
        loss = self.td_loss(data)
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        if step % self.args.trains_per_target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self) -> None:
        """Fills the replay buffer, then trains. (train_gridworld.py runs its own loop, with evaluations.)"""
        self.prepopulate_replay_buffer()
        progress_bar = tqdm(range(self.total_training_steps))
        for step in progress_bar:
            data = self.add_to_replay_buffer(self.args.steps_per_update)
            if data is not None:
                progress_bar.set_postfix(**data)
            self.training_step(step)
        self.envs.close()
