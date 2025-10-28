# Running aira-dojo on Cornell's **Unicorn** Cluster

This guide documents every change I needed to make to get
`facebookresearch/aira-dojo` working on the Cornell Unicorn cluster
(`unicorn-login-01`, `jjs533-compute-XX`).  It consolidates all of the tweaks
I made while debugging, so you can reproduce a working setup from scratch.

Useful links:
1. Akanksha's [unicorn fork of the aira-dojo repo](https://github.com/akanksha-sarkar/aira-dojo/tree/unicorn)
2. MLE-bench [repo](https://github.com/openai/mle-bench)

---
## 0. Initial Setup
- See [README](./README.md) for instructions on some initial setup for aira-dojo.
- It will be good to also follow the steps in [`src/dojo/tasks/mlebench/README.md`](./src/dojo/tasks/mlebench/README.md) to install mle-bench and run your first task. You're gonna have to install git lfs with conda (``conda install -c conda-forge git-lfs``)

## 1. Cluster prerequisites

```bash
# install Apptainer into your user conda, then source the module environment
conda install -c conda-forge apptainer
source /etc/profile.d/modules.sh
module load apptainer-1.4.0
apptainer --version          # should print: apptainer version 1.4.0+105-g938b609b6

# clone the repo somewhere under /share/<lab>/<user>/
git clone https://github.com/facebookresearch/aira-dojo.git
cd aira-dojo
```

> **Tip:** all paths in this doc use `/share/j_sun/jjk297/aira-dojo`.  Replace
> `j_sun/jjk297` with your storage location.
>
> The Unicorn build of the Apptainer superimage is already in
> `/share/j_sun/jjk297/aira-dojo/shared/sif/superimage.root.2025-05-02v2.sif`,
> so you do **not** need to rebuild it; just point `SUPERIMAGE_DIR` there in
> your `.env`.

---

## 2. Python environment

Create or reuse the project environment and install the missing wheels:

```bash
conda env create -f environment.yaml     # or conda create --name aira-dojo python=3.12
conda activate aira-dojo
pip install -e .

# PyTorch CUDA wheel that matches Unicorn's RTX 6000 Ada (CUDA 12.8 compatible)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

```

---

## 3. `.env` / `.env_default`

Copy the template and update it for Unicorn.  The key variables are:

```env
LOGGING_DIR="/share/j_sun/<user>/aira-dojo/shared/logs/"
MLE_BENCH_DATA_DIR="/share/j_sun/<user>/aira-dojo/shared/cache/dojo/tasks/mlebench/"
SUPERIMAGE_DIR="/share/j_sun/<user>/aira-dojo/shared/sif/"
TMP_BASE_DIR="/share/j_sun/<user>/aira-dojo/shared/tmp/"
USE_FAKEROOT="0"
```

I also added `GOOGLE_API_KEY=...` because I switched the solvers to the
Gemini (GDM) backend.  Hydra reads this file automatically when you launch a
run (`python -m dojo.main_run ...`).

---

## 4. Apptainer sandbox (`src/dojo/core/interpreters/jupyter/sand`)

This script bootstraps the execution environment inside the job container.
Unicorn has two limitations—no root inside Apptainer and different
filesystem layout—so I patched the script to:

* Skip overlay creation when `USE_FAKEROOT=0`.
* Copy the task dataset into `$TMP_BASE_DIR` and bind-mount it to all of:
  ` /root/data`, `/workspace/data`, and `/data`.
* Ensure the Jupyter kernel runs inside `/workspace`, while still setting up
  a writable temp dir (`TMP_BASE_DIR`) and binding the Hydra workspace there.
* Append `/etc/resolv.conf` to `APPTAINER_BIND` so DNS resolution works from
  inside the container.

These edits now ship in the repo; open the file and confirm you see the
following behaviour:

```bash
APPTAINER_BIND="${APPTAINER_BIND},${NEW}:/workspace/data:ro"
APPTAINER_BIND="${APPTAINER_BIND},${NEW}:/data:ro"
...
APPTAINER_CMD+=("bash" "-lc" "cd /workspace && ...")
```

With these changes, tasks can read `./data/...` and the visualisation exports
work again.

---

## 5. Tree visualisation fix

`src/dojo/core/solvers/utils/tree_export.py` originally used `cfg.id` when
building the export payload, which broke once Hydra rewrote the experiment
ID for logging.  Switch the line to:

```python
exp_name = cfg.exp_name
```

This is already done in our checkout (`exp_name=cfg.exp_name`).

---

## 6. Switching solvers to Gemini (GDM)

Hydra experiment configs under `src/dojo/configs/_exp/mlebench/*.yaml` now
override every solver client to `gdm` instead of the default `litellm_4o`.

Example (`aira_mcts_gdm.yaml`):

```yaml
- override /solver/client@solver.operators.draft.llm.client: gdm
```

As long as your `.env` contains `GOOGLE_API_KEY`, nothing else is required.

---

## 7. Running experiments

Example commands that now work on Unicorn:

```bash
# Greedy solver example
python -m dojo.main_run \
    +_exp=run_example \
    task.name=random-acts-of-pizza \
    logger.use_wandb=False

# MCTS solver on pizza task
python -m dojo.main_run \
    +_exp=mlebench/aira \
    task.name=random-acts-of-pizza \
    logger.use_wandb=False

# Batch the standard three MLE-Bench tasks
python run_three_tasks_local.py \
    --exp-config run_mlebench_aira_llm_mcts
```
### Defaults
- `run_three_tasks_local.py` is a thin wrapper that iterates over the three canonical MLE-Bench tasks (`aptos2019-blindness-detection`, `tabular-playground-series-may-2022`, `mlsp-2013-birds`) and seeds `[1, 2, 3]`, launching `python -m dojo.main_run ...` for each combination.
- Pass `--exp-config <name>` to swap experiments (defaults to `run_mlebench_aira_mcts_gdm`). The LLM as judge implementation is `run_mlebench_aira_llm_mcts`. UPDATE: I'm running the pmcts under `run_mlebench_aira_pmcts`

You can also configure specific tasks/seeds like so:
```
python run_three_tasks_local.py \
  --exp-config run_mlebench_aira_llm_mcts \
  --tasks aptos2019-blindness-detection tabular-playground-series-may-2022 \
  --seeds 1 2
```

Extra:
- Edit `step_limit` in the associated `.yaml` file in [solver/mlebench](src/dojo/configs/solver/mlebench)
- Edit `time_limit_secs` in the associated `.yaml` file in [solver/](src/dojo/configs/solver/)

---

## 8. Summary of repository changes

* `.env_default`, `.env`: updated paths for Unicorn and added `TMP_BASE_DIR`,
  `USE_FAKEROOT=0`.
* `requirements.txt`: add missing imports (e.g., `black`).
* `src/dojo/core/interpreters/jupyter/sand`: cluster-specific Apptainer
  binding fixes (no fakeroot overlay, `/workspace/data`).
* `src/dojo/core/solvers/utils/tree_export.py`: use `cfg.exp_name`.
* Experiment configs in `_exp/mlebench/`: override solver clients to `gdm`.
* (Optional) `src/dojo/analysis_utils/meta_error_summary.py`: skip empty
  SLURM IDs when generating error reports.

Following the steps above (hopefully) reproduces a working setup on Unicorn without
manual patch hunting.
