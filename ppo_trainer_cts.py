import numpy as np
import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor

from objectives import calc_clipped_surrogate_objective_cts, calc_entropy_bonus_cts, calc_value_function_loss
from ppo_agent_cts import PPOAgentCts
from ppo_args import PPOArgs
from ppo_trainer import PPOTrainer
from replay_minibatch import ReplayMinibatch


class PPOTrainerCts(PPOTrainer):
    def __init__(self, args: PPOArgs):
        super().__init__(args)
        self.agent = PPOAgentCts(self.envs, self.actor, self.critic, self.memory)

    def compute_ppo_objective(self, minibatch: ReplayMinibatch) -> Float[Tensor, ""]:
        """
        Changes required:
            - actor returns (mu, sigma, dist), with dist used for loss functions (rather than
                getting dist from logits)
            - objective function calculated using new `_cts` functions defined above
            - newlogprob (for logging) needs to be summed over action space
            - mu and sigma should be logged
        """
        # DISCRETE VERSION: actor returns logits, and we derive dist from them
        # dist = Categorical(logits=self.actor(minibatch.obs))
        # CONTINUOUS VERSION: actor returns distribution object directly
        mu, sigma, dist = self.agent.actor(minibatch.obs)

        values = self.agent.critic(minibatch.obs).squeeze()

        clipped_surrogate_objective = calc_clipped_surrogate_objective_cts(
            dist, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef
        )
        value_loss = calc_value_function_loss(values, minibatch.returns, self.args.vf_coef)
        entropy_bonus = calc_entropy_bonus_cts(dist, self.args.ent_coef)
        total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus

        with t.inference_mode():
            # DISCRETE VERSION: no need to sum logprobs
            # newlogprob = dist.log_prob(minibatch.actions)
            # CONTINUOUS VERSION: logprobs need to be summed over action space
            newlogprob = dist.log_prob(minibatch.actions).sum(-1)
            logratio = newlogprob - minibatch.logprobs
            ratio = logratio.exp()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        # Stash the latest loss components (incl. the policy's mu/sigma) for the progress bar + wandb.
        self.last_metrics = dict(
            value_loss=value_loss.item(),
            clip_surr=clipped_surrogate_objective.item(),
            entropy=entropy_bonus.item(),
            approx_kl=approx_kl,
            mu=mu.mean().item(),
            sigma=sigma.mean().item(),
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
