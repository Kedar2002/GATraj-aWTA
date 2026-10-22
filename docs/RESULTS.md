# aWTA experiment decision: unsuccessful

Recorded on 22 October 2026 following the user's decision to discontinue
this configuration as the proposed improvement over GATraj.

## Reported results

Values are minADE20 / minFDE20 in metres, at epoch 1,000 and seed 1.

| Held-out scene | aWTA result | Paper GATraj reference | aWTA minus paper reference |
|---|---|---|---|
| 0 ETH | 0.273403 / 0.458049 | 0.26 / 0.42 | +0.013403 / +0.038049 |
| 1 Hotel | 0.111551 / 0.164407 | 0.10 / 0.15 | +0.011551 / +0.014407 |

The earlier user-reported ETH baseline was 0.26233 / 0.41981, also better
than the aWTA result above. The user reports that the baseline is better and
has decided to record the attempt as unsuccessful.

These values were transcribed from user-provided Colab outputs, not measured
again locally. The reported runtime was Tesla T4 with PyTorch 2.11.0+cu130.
Exact recorded values and reported output paths are in [RESULTS.json](RESULTS.json).

Matched baseline JSON reports for these exact runs were not supplied, so the
reference columns must not be relabelled as a newly verified paired baseline
experiment. This is a decision to stop this configuration based on the available
evidence, not proof that every aWTA schedule fails, or a five-fold/multiple-seed
statistical result. No outcomes for Zara1, Zara2 or University are inferred.

The code and negative result remain available for reproducibility. No improved
ETH/UCY accuracy claim is made. No settings will be selected by searching these
test results for a favourable configuration.
