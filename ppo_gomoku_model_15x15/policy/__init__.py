import torch
from .ppo import PPO


def get_policy(name, cfg, action_spec, observation_spec, device="cuda"):
    if str(name).lower() != "ppo":
        raise KeyError("Only 'ppo' is available in this workspace.")
    return PPO(cfg=cfg, action_spec=action_spec, observation_spec=observation_spec, device=device)


def get_pretrained_policy(name, cfg, action_spec, observation_spec, checkpoint_path, device="cuda"):
    policy = get_policy(name=name, cfg=cfg, action_spec=action_spec, observation_spec=observation_spec, device=device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return policy
