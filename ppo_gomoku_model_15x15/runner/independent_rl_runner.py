import abc
import logging
import os
from typing import Any

import torch
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from ppo_gomoku_model_15x15.collector import SelfPlayCollector, VersusPlayCollector
from ppo_gomoku_model_15x15.env import GomokuEnv
from ppo_gomoku_model_15x15.policy import get_policy
from ppo_gomoku_model_15x15.utils.eval import eval_win_rate
from ppo_gomoku_model_15x15.utils.misc import add_prefix, set_seed
from ppo_gomoku_model_15x15.utils.policy import _policy_t, uniform_policy


class Runner(abc.ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.env = GomokuEnv(num_envs=cfg.num_envs, board_size=cfg.board_size, device=cfg.device)
        self.eval_env = GomokuEnv(num_envs=512, board_size=cfg.board_size, device=cfg.device)
        set_seed(cfg.get("seed", None))

        self.epochs = cfg.get("epochs")
        self.steps = cfg.steps
        self.save_interval = cfg.get("save_interval", -1)

        self.policy_black = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )
        self.policy_white = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )

        if black_checkpoint := cfg.get("black_checkpoint", None):
            self.policy_black.load_state_dict(torch.load(black_checkpoint, map_location=self.cfg.device))
            logging.info(f"black_checkpoint:{black_checkpoint}")
        if white_checkpoint := cfg.get("white_checkpoint", None):
            self.policy_white.load_state_dict(torch.load(white_checkpoint, map_location=self.cfg.device))
            logging.info(f"white_checkpoint:{white_checkpoint}")

        self.baseline = self._get_baseline()

        run_dir = cfg.get("run_dir", None)
        if run_dir is None:
            run_dir = wandb.run.dir
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"run_dir:{run_dir}")
        self.run_dir = run_dir

    def _get_baseline(self) -> _policy_t:
        pretrained_dir = os.path.join(
            "pretrained_models",
            f"{self.cfg.board_size}_{self.cfg.board_size}",
            f"{self.cfg.baseline.name}",
        )
        if os.path.isdir(pretrained_dir):
            ckpts = [
                p
                for f in os.listdir(pretrained_dir)
                if os.path.isfile(p := os.path.join(pretrained_dir, f)) and p.endswith(".pt")
            ]
            if ckpts:
                ckpts.sort()
                baseline = get_policy(
                    name=self.cfg.baseline.name,
                    cfg=self.cfg.baseline,
                    action_spec=self.env.action_spec,
                    observation_spec=self.env.observation_spec,
                    device=self.env.device,
                )
                logging.info(f"Baseline:{ckpts[0]}")
                baseline.load_state_dict(torch.load(ckpts[0], map_location=self.cfg.device))
                baseline.eval()
                return baseline
        return uniform_policy

    @abc.abstractmethod
    def _epoch(self, epoch: int) -> dict[str, Any]:
        ...

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if wandb.run is not None:
            wandb.run.log(info)

    def run(self, disable_tqdm: bool = False):
        pbar = tqdm(range(self.epochs), disable=disable_tqdm)
        for i in pbar:
            info = {}
            info.update(self._epoch(epoch=i))
            self._log(info=info, epoch=i)

            if i % self.save_interval == 0 and self.save_interval > 0:
                torch.save(self.policy_black.state_dict(), os.path.join(self.run_dir, f"black_{i:04d}.pt"))
                torch.save(self.policy_white.state_dict(), os.path.join(self.run_dir, f"white_{i:04d}.pt"))

            pbar.set_postfix({"fps": info["fps"]})

        torch.save(self.policy_black.state_dict(), os.path.join(self.run_dir, "black_final.pt"))
        torch.save(self.policy_white.state_dict(), os.path.join(self.run_dir, "white_final.pt"))
        self._post_run()


class SPRunner(abc.ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.env = GomokuEnv(num_envs=cfg.num_envs, board_size=cfg.board_size, device=cfg.device)
        self.eval_env = GomokuEnv(num_envs=512, board_size=cfg.board_size, device=cfg.device)
        set_seed(cfg.get("seed", None))

        self.epochs = cfg.get("epochs")
        self.steps = cfg.steps
        self.save_interval = cfg.get("save_interval", -1)

        self.policy = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )

        if checkpoint := cfg.get("checkpoint", None):
            self.policy.load_state_dict(torch.load(checkpoint, map_location=self.cfg.device))
            logging.info(f"load from {checkpoint}")

        self.baseline = self._get_baseline()

        run_dir = cfg.get("run_dir", None)
        if run_dir is None:
            run_dir = wandb.run.dir
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"run_dir:{run_dir}")
        self.run_dir = run_dir

    def _get_baseline(self) -> _policy_t:
        pretrained_dir = os.path.join(
            "pretrained_models",
            f"{self.cfg.board_size}_{self.cfg.board_size}",
            f"{self.cfg.baseline.name}",
        )
        if os.path.isdir(pretrained_dir):
            ckpts = [
                p
                for f in os.listdir(pretrained_dir)
                if os.path.isfile(p := os.path.join(pretrained_dir, f)) and p.endswith(".pt")
            ]
            if ckpts:
                ckpts.sort()
                baseline = get_policy(
                    name=self.cfg.baseline.name,
                    cfg=self.cfg.baseline,
                    action_spec=self.env.action_spec,
                    observation_spec=self.env.observation_spec,
                    device=self.env.device,
                )
                logging.info(f"Baseline:{ckpts[0]}")
                baseline.load_state_dict(torch.load(ckpts[0], map_location=self.cfg.device))
                baseline.eval()
                return baseline
        return uniform_policy

    @abc.abstractmethod
    def _epoch(self, epoch: int) -> dict[str, Any]:
        ...

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if wandb.run is not None:
            wandb.run.log(info)

    def run(self, disable_tqdm: bool = False):
        pbar = tqdm(range(self.epochs), disable=disable_tqdm)
        for i in pbar:
            info = {}
            info.update(self._epoch(epoch=i))
            self._log(info=info, epoch=i)

            if i % self.save_interval == 0 and self.save_interval > 0:
                torch.save(self.policy.state_dict(), os.path.join(self.run_dir, f"{i:04d}.pt"))

            pbar.set_postfix({"fps": info["fps"]})

        torch.save(self.policy.state_dict(), os.path.join(self.run_dir, "final.pt"))
        self._post_run()


class IndependentRLRunner(Runner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._collector = VersusPlayCollector(
            self.env,
            self.policy_black,
            self.policy_white,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )

    def _epoch(self, epoch: int) -> dict[str, Any]:
        data_black, data_white, info = self._collector.rollout(self.steps)
        info = add_prefix(info, "versus_play/")
        info["fps"] = info["versus_play/fps"]
        del info["versus_play/fps"]
        info.update(add_prefix(self.policy_black.learn(data_black.to_tensordict()), "policy_black/"))
        del data_black
        info.update(add_prefix(self.policy_white.learn(data_white.to_tensordict()), "policy_white/"))
        del data_white

        if epoch % 50 == 0 and epoch != 0:
            torch.cuda.empty_cache()
        return info

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % 5 == 0:
            info.update(
                {
                    "eval/black_vs_white": eval_win_rate(self.eval_env, player_black=self.policy_black, player_white=self.policy_white),
                    "eval/black_vs_baseline": eval_win_rate(self.eval_env, player_black=self.policy_black, player_white=self.baseline),
                    "eval/baseline_vs_white": eval_win_rate(self.eval_env, player_black=self.baseline, player_white=self.policy_white),
                }
            )
            print(
                "Black vs White:{:.2f}%\tBlack vs Baseline:{:.2f}%\tBaseline vs White:{:.2f}%".format(
                    info["eval/black_vs_white"] * 100,
                    info["eval/black_vs_baseline"] * 100,
                    info["eval/baseline_vs_white"] * 100,
                )
            )
        return super()._log(info, epoch)


class IndependentRLSPRunner(SPRunner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._collector = SelfPlayCollector(
            self.env,
            self.policy,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )

    def _epoch(self, epoch: int) -> dict[str, Any]:
        data, info = self._collector.rollout(self.steps)
        info = add_prefix(info, "self_play/")
        info["fps"] = info["self_play/fps"]
        del info["self_play/fps"]
        info.update(add_prefix(self.policy.learn(data.to_tensordict()), "policy/"))
        del data

        if epoch % 50 == 0 and epoch != 0:
            torch.cuda.empty_cache()
        return info

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % 5 == 0:
            info.update(
                {
                    "eval/player_vs_player": eval_win_rate(self.eval_env, player_black=self.policy, player_white=self.policy),
                    "eval/player_vs_baseline": eval_win_rate(self.eval_env, player_black=self.policy, player_white=self.baseline),
                    "eval/baseline_vs_player": eval_win_rate(self.eval_env, player_black=self.baseline, player_white=self.policy),
                }
            )
            print(
                "Player vs Player:{:.2f}%\tPlayer vs Baseline:{:.2f}%\tBaseline vs Player:{:.2f}%".format(
                    info["eval/player_vs_player"] * 100,
                    info["eval/player_vs_baseline"] * 100,
                    info["eval/baseline_vs_player"] * 100,
                )
            )
        return super()._log(info, epoch)
