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


hope this helps you guys out :)