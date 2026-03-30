import wandb

from omegaconf import DictConfig, OmegaConf
import datetime


def dict_flatten(a: dict, delim: str = "."):
    """Flatten a dict recursively.
    """
    result = {}
    for k, v in a.items():
        if isinstance(v, dict):
            result.update({k + delim + kk: vv for kk, vv in dict_flatten(v).items()})
        else:
            result[k] = v
    return result


def init_wandb(cfg: DictConfig):
    wandb_cfg: DictConfig = cfg.wandb
    kwargs = dict(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        name=wandb_cfg.get("name", None),
        mode=wandb_cfg.get("mode", "disabled"),
    )
    kwargs["id"] = wandb.util.generate_id()
    run = wandb.init(**kwargs)
    cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
    run.config.update(cfg_dict)
    return run
