# GATraj-aWTA

GATraj with an optional annealed winner-takes-all regression objective, adapted
from Xu et al., *Annealed Winner-Takes-All for Motion Forecasting*, ICRA 2025.
This is an independent adaptation of an existing method, not a claim of a new
loss or a measured improvement on ETH/UCY.

[Open in Colab](https://colab.research.google.com/github/Kedar2002/GATraj-aWTA/blob/main/Colab_aWTA.ipynb)

## Scope

Source baseline: https://github.com/Kedar2002/GATraj at commit `08a0042`.
The temporal encoder, social interaction computation, recurrent decoder,
20 modes and classification loss are retained. There is no RiskMotion module,
attention correction, extra model, map input, training stage or inference step.
Setting `--awta False` uses the original hard-WTA regression calculation.
Both arms receive the same runtime/resume fixes described below. Baseline
results here must be measured, not assumed to equal the paper's table.

No method can guarantee lower unseen-test ADE and FDE at fixed K=20 before
measurement. The baseline remains independently available. Optional validation
selection falls back to baseline if aWTA does not improve both metrics without
a tradeoff, but does not guarantee its decision transfers to the test scene.

## Colab workflow

Mount Drive, run setup and the CPU checks, configure a fold and run training.
The notebook follows the supplied RiskMotion notebook's layout and automatically
resumes each variant from its latest completed Drive checkpoint. By default it
trains **both** baseline and aWTA, requiring roughly twice the GPU time of one
run. Set `VARIANTS = ['awta']` to train only the modified model.

- Folds: 0 ETH, 1 Hotel, 2 Zara1, 3 Zara2, 4 University.
- Defaults: 8 observed steps, 12 predicted steps, 20 modes, 1,000 epochs,
  batch size 32, Adam learning rate 5e-4, cosine eta_min 1e-5, clip 1.
- Every completed epoch is saved directly to
  `MyDrive/GATraj-aWTA/runs/<fold>/<experiment>/<experiment>_<epoch>.tar`.
- After successful 100-epoch chunks, default pruning retains every 100th
  checkpoint and the latest. Disable pruning to retain every epoch.
- Set `STOP_AT_EPOCH=1` to check startup; keep `TOTAL_EPOCHS=1000`.
- Different flags/seeds create distinct experiment directories. Metadata checks
  reject incompatible resumes. Do not change settings to continue an old run.
- Baseline and RiskMotion checkpoints are not reused automatically. New arms
  start from the same seeded initialization and use the same batches.
- Different Colab GPU types can train different folds; compare latency on the
  same device. CPU tests do not establish CUDA or cross-device bitwise identity.

## Similar training command

```bash
python train.py --test_set 0 --seed 1 --num_epochs 1000 --epochs_per_run 100 \
  --x_encoder_layers 3 --eta_min 1e-5 --batch_size 32 --learning_rate 5e-4 \
  --randomRotate True --final_mode 20 --neighbor_thred 10 --using_cuda True \
  --gpu 0 --clip 1 --pass_time 2 --ifGaussian False --SR True \
  --input_offset True --awta True --save_base_dir ./runs
```

For baseline, change only `--awta False`. Resume a given arm with
`--load_model <last-epoch>`, keeping its full `--num_epochs 1000` unchanged.
Evaluate using the same command settings with `--phase test --load_model 1000`,
or use `evaluate.py` for JSON reports including minADE20/minFDE20, density
subsets, parameter count, label-free timing, and top-1 collision diagnostics.
Collision diagnostics are per agent-window pair at discrete future steps, not
jointly optimal scene forecasts or continuous-time collision guarantees.

## Loss adaptation

The original loss gives Laplace regression gradients only to the mode with
minimum ADE. During annealing, let d[k,i] be mode k's ADE in metres and use
`q = softmax(-d / temperature)` with detached weights. The regression loss is
the weighted sum of each mode's original per-coordinate Laplace NLL, averaged
over time, coordinates and agents. The existing soft-target classification
loss is unchanged. The Gaussian option is also supported.

Defaults anneal geometrically from `--awta_initial 0.5` to
`--awta_final 0.01` over `--awta_fraction 0.3` of the absolute epoch horizon.
For 1,000 epochs, epoch 1 uses 0.5 m, epoch 300 uses 0.01 m, and epochs
301-1000 use the original hard-WTA regression. These are unvalidated starting
settings; they are not claimed to be optimal. Temperature depends on absolute
epoch rather than process lifetime, so resuming preserves the schedule.
Evaluation predictions do not depend on annealing temperature or future labels.

## Validation and reporting

Default `--validation_set -1` is the standard four-scene training / one-scene
test protocol. For a separate pilot, set another scene index, e.g. hold out
ETH with `--test_set 0 --validation_set 4`. Both scenes are excluded from
training; the model trains on three scenes and evaluates on University.
The real ETH test data are not evaluated in that pilot. Pilot outputs are
explicitly labelled validation and stored separately. A pilot is not a
standard four-scene benchmark result.

With paired validation reports from fixed final checkpoints, `compare.py`
selects aWTA only when validation ADE and FDE are both no worse, and at least
one is strictly better; otherwise it retains baseline. Retrain the chosen
method from scratch with `--validation_set -1` on all four training scenes.
Do not choose settings or a model from test performance. Test comparison
reports both methods and their differences without selecting a winner for use.
For stronger evidence use paired seeds, e.g. 1, 2, 3, and repeat validation
across training scenes. Final reporting averages the five folds equally for
each seed and includes variation across seeds and all negative results.

## Common runtime fixes and checks

Both arms use one fixed cosine schedule and restored model, Adam, scheduler,
Python/NumPy/PyTorch RNG states. Adam is constructed after moving the model
to its device. New checkpoints use PyTorch ZIP format and a temporary file
before replacement; NumPy state uses int64 tensors. Only your own trusted
checkpoints should be loaded with `weights_only=False`. No code update can
repair a previously truncated checkpoint. Evaluation disables autograd;
forced CUDA launch blocking is removed. Preprocessing cache keys include
settings and the optional validation scene. CLI settings are explicit and
saved to YAML; saved YAML files are records, not implicit argument overrides.

Run the focused CPU checks with:

```bash
python -m unittest discover -s tests -v
```

The notebook runs them before training. They exercise loss equivalence,
mode gradients, inference label independence, schedule boundaries, exact
CPU model/Adam/scheduler/RNG continuation, ZIP checkpoints, configuration
mismatch rejection, and validation-only fallback. See
[verification record](docs/VERIFICATION.md) for actual executed checks.
Startup success is not evidence of convergence or better prediction accuracy.

## Attribution

GATraj: Mengmeng Liu et al., *A graph- and attention-based multi-agent
trajectory prediction model*, ISPRS JPRS, 2023.
[Original code](https://github.com/mengmengliu1998/GATraj),
[paper](https://doi.org/10.1016/j.isprsjprs.2023.10.001).
The inherited MIT license is retained. Dataset sources and original commands
are in [the baseline README](docs/BASE_README.md). Only ETH/UCY is implemented;
this repository does not implement nuScenes.

aWTA: Yihong Xu, Victor Letzelter, Mickael Chen, Eloi Zablocki, Matthieu Cord,
*Annealed Winner-Takes-All for Motion Forecasting*, ICRA 2025.
[Paper](https://arxiv.org/abs/2409.11172),
[official implementation](https://github.com/valeoai/MF_aWTA).
This repository implements the published assignment rule independently and
does not vendor the official UniTraj framework.
