"""The PPO objective terms, for both the discrete (Categorical) and continuous (Normal) policies."""

import torch as t
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributions.categorical import Categorical


def calc_clipped_surrogate_objective(
    dist: Categorical,
    mb_action: Int[Tensor, "minibatch_size"],
    mb_advantages: Float[Tensor, "minibatch_size"],
    mb_logprobs: Float[Tensor, "minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    """Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    dist:
        a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
    mb_action:
        what actions were taken in the sampled minibatch
    mb_advantages:
        advantages calculated from the sampled minibatch
    mb_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    """
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape
    logits_diff = dist.log_prob(mb_action) - mb_logprobs

    prob_ratio = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

    non_clipped = prob_ratio * mb_advantages
    clipped = t.clip(prob_ratio, 1 - clip_coef, 1 + clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()


def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"],
    mb_returns: Float[Tensor, "minibatch_size"],
    vf_coef: float,
) -> Float[Tensor, ""]:
    """Compute the value function portion of the loss function.

    values:
        the value function predictions for the sampled minibatch (using the updated critic network)
    mb_returns:
        the target for our updated critic network (computed as `advantages + values` from the old
        network, where `advantages` are the GAE advantages: an exponentially-weighted average, with
        weights proportional to lambda^l, of the l-step advantage estimates, each discounted by gamma).
        Note: this is not the true cumulative reward return, but a GAE-based estimate (see GAE section)
    vf_coef:
        the coefficient for the value loss, which weights its contribution to the overall loss.
        Denoted by c_1 in the paper.
    """
    assert values.shape == mb_returns.shape

    return vf_coef * (values - mb_returns).pow(2).mean()


def calc_entropy_bonus(dist: Categorical, ent_coef: float):
    """Return the entropy bonus term, suitable for gradient ascent.

    dist:
        the probability distribution for the current policy
    ent_coef:
        the coefficient for the entropy loss, which weights its contribution to the overall
        objective function. Denoted by c_2 in the paper.
    """
    return ent_coef * dist.entropy().mean()


def calc_clipped_surrogate_objective_cts(
    dist: t.distributions.Normal,
    mb_action: Float[Tensor, " minibatch_size *action_shape"],
    mb_advantages: Float[Tensor, " minibatch_size"],
    mb_logprobs: Float[Tensor, " minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    """
    Changes required:
        - logprobs need to be summed over action space
    """
    assert (mb_action.shape[0],) == mb_advantages.shape == mb_logprobs.shape

    # DISCRETE VERSION: no need to sum logprobs
    # logits_diff = probs.log_prob(mb_action) - mb_logprobs
    # CONTINUOUS VERSION: logprobs need to be summed over action space
    logits_diff = dist.log_prob(mb_action).sum(-1) - mb_logprobs

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1 - clip_coef, 1 + clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()


def calc_entropy_bonus_cts(dist: t.distributions.Normal, ent_coef: float):
    """
    Changes required:
        - entropy needs to be summed over action space before taking mean
    """
    # DISCRETE VERSION: no need to sum entropy
    # return ent_coef * dist.entropy().mean()
    # CONTINUOUS VERSION: entropy needs to be summed over action space
    return ent_coef * dist.entropy().sum(-1).mean()
