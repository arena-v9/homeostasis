import torch as t

from ppo_agent import PPOAgent


class PPOAgentCts(PPOAgent):
    def play_step(self) -> list[dict]:
        """
        Changes required:
            - actor returns (mu, sigma, dist), with dist used to sample actions
            - logprobs need to be summed over action space
        """
        obs = self.next_obs
        terminated = self.next_terminated

        with t.inference_mode():
            # DISCRETE VERSION: derive distribution from the logits
            # dist = Categorical(logits=self.actor(obs))
            # CONTINUOUS VERSION: return distribution object directly
            _, _, dist = self.actor.forward(obs)

        actions = dist.sample()

        # Step the GPU env with the action tensor directly (Brax / swing-up stay on-device)
        next_obs, rewards, next_terminated, next_truncated, infos = self.envs.step(actions)

        # DISCRETE VERSION: no need to sum logprobs
        # logprobs = dist.log_prob(actions)
        # CONTINUOUS VERSION: logprobs need to be summed over action space
        logprobs = dist.log_prob(actions).sum(-1)

        with t.inference_mode():
            values = self.critic(obs).flatten()
        self.memory.add(obs, actions, logprobs, values, rewards, terminated)

        self.next_obs = next_obs
        self.next_terminated = next_terminated.float()

        self.step += self.envs.num_envs
        return infos
