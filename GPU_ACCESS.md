Full guide here: https://tsg.cs.ucl.ac.uk/timeshare-gpus/


But if you are having difficulties then follow the instructions below:


Need to be in UCL CS Network.

Use external gateway with your normal UCL account by running:

```bash
ssh <user>@ssh-gateway.ucl.ac.uk
```
>(e.g. ssh zcabtem@ssh-gateway.ucl.ac.uk)
>(e.g. ssh zcabkas@ssh-gateway.ucl.ac.uk)


Once in external gateway, access internal CS gateway machine with your CS account:

```bash
ssh <cs-account>@knuckles.cs.ucl.ac.uk
```
>(e.g. ssh tmoody@knuckles.cs.ucl.ac.uk)
>(e.g. ssh tkasayap@knuckles.cs.ucl.ac.uk)


Once in internal CS gateway machine, access the timeshare machine with your CS account:

```bash
ssh <cs-account>@<machine>.cs.ucl.ac.uk
```
>(e.g. ssh tmoody@cream.cs.ucl.ac.uk)
>(e.g. ssh tkasayap@vanilla.cs.ucl.ac.uk)

Available machines include:
- blaze: 4x Titan X cards, 64GB RAM
- cream: 4x Quadro RTX 6000 cards, 375GB RAM
- vanilla: 4x Quadro RTX 6000 cards, 375GB RAM


Once in timeshare machine, make sure your connection only dedicates itself to a single GPU by running:

```bash
source /usr/local/cuda/CUDA_VISIBILITY.csh
```

Create a temporary cache (will get wiped when you log out):

```bash
mkdir -p /tmp/<cs-user>_uv_cache
```
>(e.g. mkdir -p /tmp/tmoody_uv_cache)
>(e.g. mkdir -p /tmp/tkasayap_uv_cache)


On first setup, initialise startup:

```bash
nano ~/.uclcs-csh-aliases
```

and paste the following:

```bash
# Ensure the UV cache directory exists on this specific machine
mkdir -p /tmp/<cs-user>_uv_cache


# Set environment variables
setenv PATH "$HOME/.local/bin:$PATH"
setenv UV_CACHE_DIR "/tmp/<cs-user>_uv_cache"

# Assign exactly one GPU
source /opt/cuda/scripts/CUDA_VISIBILITY.csh
```

example:
```bash 
# Ensure the UV cache directory exists on this specific machine
mkdir -p /tmp/tmoody_uv_cache

# Set environment variables
setenv PATH "$HOME/.local/bin:$PATH"
setenv UV_CACHE_DIR "/tmp/tmoody_uv_cache"

# Assign exactly one GPU
source /opt/cuda/scripts/CUDA_VISIBILITY.csh

ulimit -n 65535
```

Save and Exit
Ctrl + O, then Enter to save.

Ctrl + X to exit.


activate these changes with: 

```bash
source ~/.uclcs-csh-aliases
```


Ensure you are in your local files by running
```bash 
ls
```

and you should see files along the line of "core perl5 thunderbird win32 WINDOWS"



Clone the repository:
```bash
git clone https://github.com/tara-kas/RL-Coursework
```

Enter the repository with:
```bash
cd RL-Coursework
```

Install pip into python by running:
```bash
python -m pip install uv
```

You can then initialise the project with:
```bash
uv venv .venv --python 3.11
```
> you may need to re run "source ~/.uclcs-csh-aliases"

Sync your venv with
```bash
uv sync
```

Add the following packages to your uv environment with
```bash
uv add torch torchvision torchaudio
```

Increase the number of terminals you can have open at once:
```bash
ulimit -n 65535
```

You can now run any of the regular commands e.g.:
```bash
uv run python train.py --board_size 9 --agent_type alphazero --amp --num_workers 1 --worker_device cuda --no-compile --mcts_batch_size 64 --batch_size 128 --iterations 300 --games_per_iteration 200 --eval_games 200 --learning_rate 2e-4 --num_simulations 200 --value_coef 2.5 --c_puct 2.0 --self_play_temp 1.0 --temp_moves 30 --league_prob 0.2 --heuristic_prob 0.4 --az_best_by heuristic --az_eval_freq 10 --az_eval_games_best 200 --root_dirichlet_alpha 0.3 --root_dirichlet_epsilon 0.25
```



hope this helps you guys out :)