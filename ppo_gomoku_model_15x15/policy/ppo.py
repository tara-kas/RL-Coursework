import torch
from torch.optim import Adam, AdamW
from torchrl.modules import ProbabilisticActor, ValueOperator, ActorValueOperator, SafeModule
from torch.distributions.categorical import Categorical
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate
from ppo_gomoku_model_15x15.utils.module import ValueNet, ActorNet, ResidualTower, PolicyHead, ValueHead


def make_ppo_actor(cfg, action_spec, device):
    actor_net = ActorNet(
        residual_tower=ResidualTower(
            in_channels=3,
            num_channels=cfg.num_channels,
            num_residual_blocks=cfg.num_residual_blocks,
        ),
        out_features=action_spec.space.n,
        num_channels=cfg.num_channels,
    ).to(device)

    policy_module = TensorDictModule(module=actor_net, in_keys=["observation", "action_mask"], out_keys=["probs"])
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["probs"],
        distribution_class=Categorical,
        return_log_prob=True,
    )
    return policy_module


def make_critic(cfg, device):
    value_net = ValueNet(
        residual_tower=ResidualTower(
            in_channels=3,
            num_channels=cfg.num_channels,
            num_residual_blocks=cfg.num_residual_blocks,
        ),
        num_channels=cfg.num_channels,
    ).to(device)
    value_module = ValueOperator(module=value_net, in_keys=["observation"])
    return value_module


def make_ppo_ac(cfg, action_spec, device):
    residual_tower = ResidualTower(
        in_channels=3,
        num_channels=cfg.num_channels,
        num_residual_blocks=cfg.num_residual_blocks,
    ).to(device)
    common_module = SafeModule(module=residual_tower, in_keys=["observation"], out_keys=["hidden"])

    policy_head = PolicyHead(out_features=action_spec.space.n, num_channels=cfg.num_channels).to(device)
    policy_module = TensorDictModule(module=policy_head, in_keys=["hidden", "action_mask"], out_keys=["probs"])
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["probs"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    value_head = ValueHead(num_channels=cfg.num_channels).to(device)
    value_module = ValueOperator(module=value_head, in_keys=["hidden"])
    return ActorValueOperator(common_module, policy_module, value_module)


def make_dataset_naive(tensordict, batch_size):
    tensordict = tensordict.reshape(-1)
    assert tensordict.shape[0] >= batch_size
    perm = torch.randperm((tensordict.shape[0] // batch_size) * batch_size, device=tensordict.device).reshape(-1, batch_size)
    for indices in perm:
        yield tensordict[indices]


def get_optimizer(cfg, params):
    options = {"adam": Adam, "adamw": AdamW}
    name = cfg.name.lower()
    assert name in options
    return options[name](params=params, **cfg.kwargs)


class PPO:
    def __init__(self, cfg, action_spec, observation_spec, device="cuda"):
        self.cfg = cfg
        self.device = device

        self.clip_param = cfg.clip_param
        self.ppo_epoch = int(cfg.ppo_epochs)

        self.entropy_coef = cfg.entropy_coef
        self.gae_gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.average_gae = cfg.average_gae
        self.batch_size = int(cfg.batch_size)

        self.max_grad_norm = cfg.max_grad_norm
        if self.cfg.get("share_network"):
            actor_value_operator = make_ppo_ac(cfg, action_spec=action_spec, device=self.device)
            self.actor = actor_value_operator.get_policy_operator()
            self.critic = actor_value_operator.get_value_head()
        else:
            self.actor = make_ppo_actor(cfg=cfg, action_spec=action_spec, device=self.device)
            self.critic = make_critic(cfg=cfg, device=self.device)

        fake_input = observation_spec.zero()
        fake_input["action_mask"] = ~fake_input["action_mask"]
        with torch.no_grad():
            self.actor(fake_input)
            self.critic(fake_input)
        # print(f"actor params:{count_parameters(self.actor)}")
        # print(f"critic params:{count_parameters(self.critic)}")

        self.loss_module = ClipPPOLoss(
            actor=self.actor,
            critic=self.critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            normalize_advantage=self.cfg.get("normalize_advantage", True),
            loss_critic_type="smooth_l1",
        )

        self.optim = get_optimizer(self.cfg.optimizer, self.loss_module.parameters())

    def __call__(self, tensordict):
        tensordict = tensordict.to(self.device)
        actor_input = tensordict.select("observation", "action_mask", strict=False)
        actor_output = self.actor(actor_input)
        actor_output = actor_output.exclude("probs")
        tensordict.update(actor_output)

        # share_network=True uses "hidden" as critic input.
        # share_network=False uses "observation" as critic input.
        critic_input = tensordict.select("hidden", "observation", strict=False)
        critic_output = self.critic(critic_input)
        tensordict.update(critic_output)

        return tensordict

    def learn(self, data):
        # Compute GAE per batch.
        value = data["state_value"].to(self.device)
        next_value = data["next", "state_value"].to(self.device)
        done = data["next", "done"].unsqueeze(-1).to(self.device)
        reward = data["next", "reward"].to(self.device)
        with torch.no_grad():
            adv, value_target = vec_generalized_advantage_estimate(
                self.gae_gamma,
                self.gae_lambda,
                value,
                next_value,
                reward,
                done=done,
                terminated=done,
                time_dim=data.ndim - 1,
            )
            loc = adv.mean()
            scale = adv.std().clamp_min(1e-4)
            if self.average_gae:
                adv = adv - loc
                adv = adv / scale

            data.set("advantage", adv)
            data.set("value_target", value_target)

        # Filter invalid white transitions.
        invalid = data.get("invalid", None)
        if invalid is not None:
            data = data[~invalid]

        data = data.reshape(-1)

        self.train()
        loss_objectives = []
        loss_critics = []
        loss_entropies = []
        losses = []
        grad_norms = []
        for _ in range(self.ppo_epoch):
            for minibatch in make_dataset_naive(data, batch_size=self.batch_size):
                minibatch = minibatch.to(self.device)
                loss_vals = self.loss_module(minibatch)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss_objectives.append(loss_vals["loss_objective"].clone().detach())
                loss_critics.append(loss_vals["loss_critic"].clone().detach())
                loss_entropies.append(loss_vals["loss_entropy"].clone().detach())
                losses.append(loss_value.clone().detach())
                # Optimisation step: backprop, gradient clip, optimiser step.
                loss_value.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(), self.max_grad_norm
                )
                grad_norms.append(grad_norm.clone().detach())
                self.optim.step()
                self.optim.zero_grad()

        self.eval()
        return {
            "advantage_meam": loc.item(),
            "advantage_std": scale.item(),
            "grad_norm": torch.stack(grad_norms).mean().item(),
            "loss": torch.stack(losses).mean().item(),
            "loss_objective": torch.stack(loss_objectives).mean().item(),
            "loss_critic": torch.stack(loss_critics).mean().item(),
            "loss_entropy": torch.stack(loss_entropies).mean().item(),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.critic.load_state_dict(state_dict["critic"], strict=False)
        self.actor.load_state_dict(state_dict["actor"])

        self.loss_module = ClipPPOLoss(
            actor=self.actor,
            critic=self.critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            normalize_advantage=self.cfg.get("normalize_advantage", True),
            loss_critic_type="smooth_l1",
        )

        self.optim = get_optimizer(self.cfg.optimizer, self.loss_module.parameters())

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
