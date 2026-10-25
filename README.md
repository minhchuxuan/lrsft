# LR-SFT prototype: Layer-Reserved SFT for RL-ready reasoning models

Pick the few decoder layers that GRPO relies on most at the base model θ₀ (the reserved set R),
freeze them during SFT, then run RL on R only. Comparing this with plain SFT followed by the same
RL on the same R asks whether the RL headroom lost after deep SFT sits in those parameters or in
the output distribution.

Scaled-down setting: **Qwen3-0.6B-Base** (28 layers) on **1× H100 80GB**. The 1.7B / 4B runs and
the AESL arm of the proposal are future work.

| Step | What it does | Main output |
|---|---|---|
| A1 | θ₀: 8 rollouts on each of 768 fixed MATH-train prompts (all levels), keep the informative groups (not all-same reward), GRPO gradient per layer on 3 batches of 64 groups; R = top-2 | `results/probe/base.json`, `results/layers.json` |
| A2 | Full SFT and LR-SFT (R frozen), same data, seed and order; snapshots at epochs 1, 3, 8; bit-identity check of R | `outputs/models/{full,lr}_ep{1,3,8}`, `results/verify_lr.json` |
| A3 | Same probe at every SFT checkpoint: Spearman ρ and top-k overlap with θ₀ | `results/summary_checkpoints.csv`, `fig_a3_layer_ranking.png` |
| A4 | Pre-RL diagnostics: MATH-500 pass@1 / pass@k (unbiased), token entropy, dead-group rate, CKA of R's input vs θ₀, weight drift | `fig_a4_*.png` |
| B | GRPO (DAPO tricks) on R only, from full-SFT and LR-SFT checkpoints, deepest first; MATH-500 after RL, peak memory, wall-clock, rollout tokens | `results/summary_rl.csv`, `fig_b_rl_vs_depth.png` |
| extras | At the deepest full-SFT checkpoint: random R, R′ re-selected after SFT, full-parameter GRPO | `fig_b_controls_cost.png` |

## Running on the H100

```bash
# 1. copy the repo to the machine, then (once, ~5 min):
./setup.sh                        # uv venv + pinned torch/vllm/transformers/trl; picks CUDA 13.0 or 12.9 from the driver

# 2. 15-25 min end-to-end check with the real model, vLLM and TRL colocate (separate output dirs):
CONFIGS="configs/default.yaml configs/gpu_smoke.yaml" ./run_all.sh

# 3. the real run, unattended (resumes if re-run after a crash or reboot):
tmux new -d -s lrsft ./run_all.sh
#    or only part A first:            STAGES=A ./run_all.sh
#    or the cheaper profile:          CONFIGS="configs/default.yaml configs/fast.yaml" ./run_all.sh
```

Watching it:

```bash
tail -f logs/run_all_*.log                     # one line per step
tail -f logs/<step>.log                         # full output of one step, e.g. logs/sft_full.log
cat results/pipeline_status.json                # state and seconds per step
python -m lrsft.pipeline --config configs/default.yaml --dry-run    # plan: done / ready / waiting
python plot.py --config configs/default.yaml    # tables + figures from whatever exists so far
```

- **Idempotent.** Every step declares its output files and is skipped when they exist. SFT and
  RL also resume from their own Trainer checkpoints. A failing step is retried once; the pipeline
  then continues with the steps that do not depend on it. A lock file stops a second pipeline from
  starting on the same `results/` dir.
- **One process per step**, so vLLM (sampling) and HF (training/probing) never share GPU memory
  except inside GRPO, where TRL colocates them on purpose.
- **Caches.** `HF_HOME` defaults to `./.hf_cache`; set it before `run_all.sh` to reuse an existing
  cache. vLLM / Triton / Inductor compile caches go to `./.cache`. `VENV=/path/to/venv` (both
  scripts) and `UV_INSTALL_DIR`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` (setup.sh) keep the install
  wherever you want; setup.sh never edits shell rc files. Big artifacts go to `outputs/` (~45 GB at the default size; `LRSFT_WORK=/path` moves them).
  Small results (JSON, CSV, PNG) go to `results/`.
- **Early results.** After `prep → probe θ₀ → full SFT → probes at ep1/3/8` (≈3 h) the step
  `report__A3` already draws the A3 figure. All of part A takes ≈6–7 h.
- **Scaling.** Every size is a config value: `--set data.sft_num_samples=10000 rl.rounds=40
  rl.depths=[8,1] eval.n=8`, or stack `configs/fast.yaml`. `select.override=[i,j]` forces R.
- **Small debug GPU.** `configs/small_gpu.yaml` (fp16 snapshots, fp32 training, smaller sizes) is for
  a pre-Ampere card without bf16. On a 6 GB GTX 1660 SUPER the whole pipeline ran with the tiny model
  (vLLM sampling, SFT, verify, GRPO colocate on R / random R / R′ / all params), and with the real
  Qwen3-0.6B: vLLM sampling, the layer probe, diagnostics and one GRPO-on-R round with vLLM
  colocated (5.2 GB peak; vLLM vs HF log-prob gap 0.001). Full SFT of 0.6B with Adam does not fit
  there; never use that config for results.
- **Redoing one step.** Delete its output file(s) (names in `lrsft/pipeline.py`), then run
  `./run_all.sh --only <step>` (e.g. `--only eval__full_ep8`). Changing a config value does not
  invalidate finished steps by itself.

## GPU-hour estimate (1× H100, default config)

Measured on the real data (prep run locally): 49.7k OpenR1 problems pass the filters, the 15k kept
average 2.6k tokens (p90 3.75k), i.e. 39M tokens per epoch; MATH train levels 3–5 gives 5,582 RL
prompts. Assumptions: SFT'd models write ≈2.5k-token answers, the base
model ≈0.6k; HF training at ≈30% MFU; vLLM ≈20–30k tokens/s for a 0.6B model. Treat these as
±50% until the GPU smoke run gives real numbers (`results_smoke/pipeline_status.json`).

| Stage | Work | H100-h |
|---|---|---|
| prep | download OpenR1-Math / MATH / MATH-500, filter, tokenize (CPU) | 0.3 |
| A1 probe θ₀ + eval θ₀ | 4k rollouts + 3×64 group backward; MATH-500 ×16 | 0.3 |
| A2 full SFT | 15k samples × 8 epochs ≈ 300M tokens | 1.5–2 |
| A2 LR-SFT | same | 1.5–2 |
| A3 probes | 6 checkpoints × (6k rollouts ≈15M tokens + gradients) | 1.5 |
| A4 eval + diagnostics | 6 checkpoints × MATH-500 ×16 (≈20M tokens) | 1.5 |
| **A total** | | **≈7** |
| B | 6 GRPO runs × 60 rounds × 512 rollouts (≈2 min/round) + eval | ≈13 |
| extras | 3 GRPO runs (random R, R′, all params) + eval | ≈6.5 |
| **A + B + extras** | | **≈27** |

`configs/fast.yaml` (10k SFT samples, eval n=8, 40 rounds, depths 8 and 1, two controls): ≈14 h.

## Stack and why

| Package | Pin | Reason |
|---|---|---|
| vLLM | 0.31.0 | Newest release TRL 1.16 accepts (`vllm<=0.31.0`). Used for every sampling job and inside GRPO. |
| torch | 2.13.0 | Hard-pinned by vLLM 0.31.0. PyPI wheels are CUDA 13.0 (driver ≥580); `setup.sh` switches to the `+cu129` wheels of torch and vLLM on older drivers. |
| transformers | 5.17.0 | vLLM 0.31.0 needs `<5.18`. |
| TRL | 1.16.0 | `GRPOTrainer` has everything needed: `loss_type="dapo"` (token-level loss), `epsilon_high` (clip-higher), `mask_truncated_completions`, vLLM colocate on one GPU, a chunked LM head (no full logits), and importance-sampling correction for the vLLM/HF mismatch. `SFTTrainer` gives completion-only loss and logs entropy. |
| accelerate / datasets | 1.15.0 / 5.1.0 | current releases compatible with the above |
| math-verify | 0.9.0 | answer checking (with latex2sympy2_extended 1.11.0, antlr4 4.13.2) |
| flash-attn | not used | HF side runs SDPA; vLLM ships its own attention kernels. Avoids a fragile source build against torch 2.13. |

Why TRL and not verl: one GPU, a 0.6B model, and two days. TRL colocate runs in one Python
process with the model object we build ourselves, so freezing is just `requires_grad=False` and
the optimizer automatically sees only R. verl's actor/rollout workers, Ray and FSDP configs add a
lot of surface for no gain at this scale. These APIs were checked against the source of the pinned
versions (TRL 1.16.0 `grpo_trainer.py`, `vllm_generation.py`, `sft_trainer.py`; transformers 5.17.0
`Trainer`; vLLM 0.31.0 `SamplingParams` / `LLM`).

## Design decisions and pitfalls handled

- **Freezing.** `requires_grad=False` on `model.layers[i]` for i ∈ R (LR-SFT), or on everything
  except R (RL). HF `Trainer.create_optimizer` keeps only `requires_grad` params; `rl.py` asserts
  the optimizer holds exactly the trainable count, and after RL checks that every frozen tensor is
  unchanged. `verify.py` checks that R in each LR-SFT snapshot is **bit-identical** to θ₀ on disk.
- **Precision.** Weights are fp32 masters with bf16 autocast. bf16 weights would silently drop
  RL-sized updates (lr 1e-6), and bf16→fp32 is lossless, so frozen layers stay bit-exact.
  Snapshots are saved in bf16 (`config.dtype=bfloat16`, which is what vLLM picks).
- **Gradient checkpointing** always with `use_reentrant=False` (the reentrant form can drop
  gradients when a block's inputs do not require grad). A test checks that grads with frozen
  blocks and checkpointing equal grads without checkpointing.
- **Tied embeddings** (Qwen3-0.6B). The embedding is the LM head. It trains in both SFT arms and
  is frozen in RL-on-R. Snapshots are rebuilt from the config so the tie survives saving (tested).
- **One prompt format everywhere** (`lrsft/common.py`): ChatML user turn + "put your final answer
  within \boxed{}", completion ends with `<|im_end|>`. SFT examples are tokenized as
  `tok(prompt) + tok(trace + <|im_end|>)`, so the prompt tokens equal what GRPO and eval feed.
- **Stop tokens.** The base model's eos is `<|endoftext|>`. Every saved model gets
  `generation_config.eos_token_id = [<|endoftext|>, <|im_end|>]` (on a real run the base model ended
  its answers with `<|endoftext|>`, never `<|im_end|>`, so both are needed); TRL reads this list to decide
  which completions are truncated (otherwise every `<|im_end|>` answer would be masked as truncated),
  and vLLM gets the same list as `stop_token_ids`.
- **vLLM start method.** Sampling jobs force `VLLM_WORKER_MULTIPROC_METHOD=spawn`; with the default
  fork the engine process died because CUDA was already initialised in the parent (seen on Turing).
- **vLLM weight sync.** In colocate mode TRL pushes all parameters to vLLM after each update
  (0.6B, cheap); tied weights are yielded once. Frozen layers are therefore always consistent.
- **Reward.** Last `\boxed{}` of the completion vs the gold answer with math-verify (both wrapped
  in `$…$`, default normalisation). Same function for probes, GRPO and eval.
- **Layer probe = the GRPO gradient at ratio 1.** Per prompt group, the token-mean DAPO loss
  −Σᵢ Aᵢ Σₜ log π(oᵢₜ) / Σ|oᵢ| (TRL's advantage formula, truncated completions masked) is
  back-propagated once; grad_norm = ‖mean of group gradients‖ per layer, plus the group-level
  Fisher trace E‖g_j‖² and the SNR. All 28 layers have the same shape, so dividing by the parameter
  count does not change the ranking; `rel_grad = ‖g‖/‖W‖` is reported too. No weights are updated
  or kept. R′ (re-selected after SFT) uses the same code at the deepest full-SFT checkpoint.
- **SFT depth.** Constant LR after a 3% warmup, so "epoch 1 / 3 / 8" means more training, not a
  different point of a decay schedule. Same seed and data order for both arms.
- **RL config.** 64 prompts × 8 rollouts per round, 4 optimizer updates of 128 rollouts per round
  (so clip-higher actually acts), ε=0.2 / ε_high=0.28, no KL, max 4096 new tokens, lr 5e-6 for
  R-only and 1e-6 for full parameters. The vLLM/HF log-prob mismatch is corrected per token
  (`token_truncate`, ratio ≤ 3) instead of TRL's default `sequence_mask`, which sums the log-ratio
  over a whole answer and can drop long 4k-token answers. DAPO dynamic sampling is not in TRL; the
  dead-group rate (`frac_reward_zero_std`) is logged per step instead.
- **Data.** SFT: OpenR1-Math-220k (default split), complete and math-verify-correct R1 traces,
  prompt + trace ≤ 4096 tokens, shortest correct trace per problem, 15k samples. RL: MATH train
  levels 3–5 (DAPO-Math-17k is available as `data.rl_source=dapo`, but it is hard for 0.6B and would
  mostly give dead groups). Probe pool: 768 fixed MATH-train prompts of all levels, the same for
  every checkpoint; on a debug run with level 3–5 prompts, 78% of θ₀'s groups were all-wrong, too
  few informative groups for three batches. Eval: MATH-500. Problems that appear in MATH-500 are
  removed from SFT and RL data.
- **Seeds.** One `seed` in the config drives data shuffles, sampler order, the random-R control and
  vLLM sampling.

## Tests (CPU, on a Mac)

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
pytest -q -m "not slow"     # 41 unit tests, ~20 s
pytest -q -m slow           # whole pipeline on a random 4-layer Qwen3 + synthetic data (~6 min)
```

They cover: freezing and optimizer contents, gradients through frozen blocks with checkpointing,
LR-SFT keeping R bit-identical (in memory and in the bf16 snapshot), the SFT loss mask, tied
embeddings after saving, chunked log-probs and entropy vs a full softmax, the probe statistics
recomputed by hand, micro-batch invariance, truncation masking, the reward on tricky answers,
pass@k vs brute force, dead groups, Spearman and CKA invariances. The slow test runs every stage
through `lrsft.pipeline` with `configs/tiny.yaml`: HF `generate` instead of vLLM, TRL GRPO without
vLLM, and a random-but-deterministic reward. vLLM itself is only exercised by the GPU smoke run.

## Layout

```
configs/      default.yaml (H100), fast.yaml (time-boxed), gpu_smoke.yaml, tiny.yaml (CPU)
lrsft/
  config.py        YAML + --set overrides, all paths
  common.py        prompt format, stop tokens, load / freeze / save_snapshot, IO
  data.py          prep: base model, SFT / RL / probe / MATH-500 data
  rollout.py       sampling (vLLM or HF) + scoring; probe rollouts, MATH-500 pass@k
  reward.py        \boxed{} extraction + math-verify
  scoring.py       chunked log-probs / entropy, GRPO advantages, per-layer gradient statistics
  probe.py         A1 / A3 layer ranking
  select_layers.py R, random control, R'
  sft.py           A2 full / LR SFT (TRL SFTTrainer)
  verify.py        bit-identity of R in LR-SFT snapshots
  diagnostics.py   A4 entropy, CKA of layer inputs vs θ₀, weight drift
  rl.py            B GRPO on R / all params (TRL GRPOTrainer, vLLM colocate)
  metrics.py       pass@k, dead groups, Spearman, top-k overlap, linear CKA
  pipeline.py      stage graph, skip-if-done, retries, status
plot.py       summary CSVs + slide figures
setup.sh, run_all.sh
tests/
```

## Known limits

- Not yet run on a GPU: the vLLM paths and GRPO colocate memory are untested until the smoke run.
- R-only GRPO learning rate (5e-6) is a guess (≈5× the full-parameter lr to compensate for
  training ~5% of the weights); if rewards barely move in `results/rl/*_log.jsonl`, raise it.
- One seed per cell; differences of 1–2 points on MATH-500 are within noise.
- Freezing R keeps its weights, not its input distribution or downstream Jacobian (as stated in
  the proposal); the CKA and entropy diagnostics are there to show how much of that changed.
