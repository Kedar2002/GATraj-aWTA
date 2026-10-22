# Verification record

Executed locally on 21 October 2026 using Windows, Python 3.10 and CPU
PyTorch 2.6.0. This is a startup and correctness record, not a performance
claim or an executed Colab/CUDA verification.

## Focused checks

All seven CPU checks passed:

1. Absolute-epoch temperature boundary and hard-WTA transition.
2. Detached assignment, nonzero regression gradients for multiple modes and
   low-temperature hard assignment limit.
3. Same seeded parameter initialization in both variants, exact original
   baseline loss formula, equal predictions for identical weights, finite
   training gradients, and inference independence from future labels.
4. Gaussian hard-WTA loss limit and finite Laplace/Gaussian small-scale losses.
5. Distinct experiment names for variant, schedule and validation split.
6. Exact continuation of CPU model weights and scheduler after saving and
   reloading Adam and Python/NumPy/PyTorch RNG states. ZIP checkpoint output,
   temporary-file replacement and incompatible configuration rejection.
7. Validation-only fallback to baseline on an ADE/FDE tradeoff, plus no
   automatic selection from test results.

## Actual data startup and resume

Both baseline and aWTA independently trained epoch 1, saved, loaded epoch 1
in a new process, and trained epoch 2. Both then ran `evaluate.py` successfully.
The full learning-rate and annealing horizon stayed at 1,000 epochs throughout;
`epochs_per_run=1` was used for the check.

The startup split excluded Hotel (outer held-out test scene, ID 1) and
University (designated validation scene, ID 4) from training. Training used
83 batches from the other three scenes. Evaluation used 14 validation batches
containing 14,295 agent-windows. The Hotel test scene was not evaluated.
Both reports were labelled `validation`, and both models had 293,131 parameters.

No startup weights or early accuracy numbers are published as benchmark
results. No 1,000-epoch run, multiple-seed accuracy study, CUDA execution,
Drive outage simulation, or improvement over the baseline has been established.

The Colab notebook JSON, all code cells and Python source parse successfully;
notebook execution outputs are cleared. Its setup cell repeats the focused
CPU checks in the Colab runtime before starting GPU training.

Subsequent user-reported 1,000-epoch Colab results are documented separately
in [RESULTS.md](RESULTS.md); they did not demonstrate an improvement.
