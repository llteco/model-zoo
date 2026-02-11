from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Union

import torch
import torch.distributions.constraints
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.data import Composite, UnboundedContinuous
from torchrl.envs.transforms import CatTensors
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor


@dataclass
class FeatureExtractorConfig:
    learning_rate: float = 5e-4
    dyn_obs_num: int = 5


@dataclass
class ActorConfig:
    learning_rate: float = 5e-4
    clip_ratio: float = 0.1
    action_limit: float = 2.0  # m/s


@dataclass
class CriticConfig:
    learning_rate: float = 5e-4
    clip_ratio: float = 0.1


@dataclass
class AlgoConfig:
    feature_extractor: FeatureExtractorConfig = field(
        default_factory=FeatureExtractorConfig
    )
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)

    entropy_loss_coefficient: float = 1e-3
    training_frame_num: int = 32
    training_epoch_num: int = 4
    num_minibatches: int = 16


cfg = AlgoConfig()


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


def vec_to_new_frame(vec, goal_direction):
    if len(vec.size()) == 1:
        vec = vec.unsqueeze(0)
    # print("vec: ", vec.shape)

    # goal direction x
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.0], device=vec.device, dtype=vec.dtype)

    # goal direction y
    goal_direction_y = torch.cross(
        z_direction.expand_as(goal_direction_x), goal_direction_x
    )
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)

    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(
            vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)
        )
        vec_y_new = torch.bmm(
            vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1)
        )
        vec_z_new = torch.bmm(
            vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1)
        )
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1.0, 0, 0], device=vec.device, dtype=vec.dtype).expand_as(
        goal_direction
    )

    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coodirnate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel


class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()

    def forward(self, features: torch.Tensor):
        alpha = 1.0 + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1.0 + self.beta_softplus(self.beta_layer(features)) + 1e-6
        return alpha, beta


class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {
        "alpha": torch.distributions.constraints.positive,
        "beta": torch.distributions.constraints.positive,
    }

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)


class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor

    def forward(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor,
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step]
                + self.gamma * next_value[:, step] * not_done[:, step]
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (
                self.gamma * self.lmbda * not_done[:, step] * gae
            )
        returns = advantages + value
        return advantages, returns


class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta
        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out


class PPO(TensorDictModuleBase):
    def __init__(self, observation_spec, action_spec):
        super().__init__()
        self.cfg = cfg

        # Feature extractor for LiDAR
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]),
            nn.ELU(),
            nn.LazyConv2d(
                out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]
            ),
            nn.ELU(),
            nn.LazyConv2d(
                out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]
            ),
            nn.ELU(),
            nn.Flatten(1),
            nn.LazyLinear(128),
            nn.LayerNorm(128),
        )

        # Dynamic obstacle information extractor
        dynamic_obstacle_network = nn.Sequential(
            nn.Flatten(1),
            make_mlp([128, 64]),
        )

        # Feature extractor
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(
                feature_extractor_network,
                [("agents", "observation", "lidar")],
                ["_cnn_feature"],
            ),
            TensorDictModule(
                dynamic_obstacle_network,
                [("agents", "observation", "dynamic_obstacle")],
                ["_dynamic_obstacle_feature"],
            ),
            CatTensors(
                [
                    "_cnn_feature",
                    ("agents", "observation", "state"),
                    "_dynamic_obstacle_feature",
                ],
                "_feature",
                del_keys=False,
            ),
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        )

        # Actor etwork
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(
                BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]
            ),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True,
        )

        # Critic network
        self.critic = TensorDictModule(nn.LazyLinear(1), ["_feature"], ["state_value"])
        self.value_norm = ValueNorm(1)

        # Loss related
        self.gae = GAE(0.99, 0.95)  # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(
            delta=10
        )  # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(
            self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate
        )
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor.learning_rate
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.actor.learning_rate
        )

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)

        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_)
        self.critic.apply(init_)

    def forward(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world
        actions = (
            2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit
        ) - self.cfg.actor.action_limit
        actions_world = vec_to_world(
            actions, tensordict["agents", "observation", "direction"]
        )
        tensordict["agents", "action"] = actions_world
        return tensordict


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = self.init_model()

    # PPO policy loader
    def init_model(self):
        observation_dim = 8
        observation_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": Composite(
                            {
                                "state": UnboundedContinuous((observation_dim,)),
                                "lidar": UnboundedContinuous((1, 36, 4)),
                                "direction": UnboundedContinuous((1, 3)),
                                "dynamic_obstacle": UnboundedContinuous((1, 5, 10)),
                            }
                        ),
                    }
                ).expand(1)
            },
            shape=[1],
        )

        action_dim = 3
        action_spec = Composite(
            {
                "agents": Composite(
                    {
                        "action": UnboundedContinuous((action_dim,)),
                    }
                )
            }
        ).expand(1, action_dim)

        policy = PPO(observation_spec, action_spec)
        return policy

    def forward(self, robot_state, static_obs_input, dyn_obs_input, target_dir):
        obs = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": TensorDict(
                            {
                                "state": robot_state,
                                "lidar": static_obs_input,
                                "direction": target_dir,
                                "dynamic_obstacle": dyn_obs_input,
                            }
                        )
                    }
                )
            },
        )

        with set_exploration_type(ExplorationType.MEAN):
            output = self.policy(obs)
            velocity = output["agents", "action"][0][0][:2]
        return velocity
