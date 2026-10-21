"""Compare fixed checkpoints. Choose a variant only from whole-scene validation."""
import argparse
import json
from pathlib import Path


def compare(baseline, modified):
    a, b = baseline['experiment'], modified['experiment']
    ignored = {'awta', 'awta_initial', 'awta_final', 'awta_fraction', 'train_model',
               'model_dir', 'config', 'model_save_path', 'save_base_dir', 'save_dir', 'epochs_per_run'}
    mismatches = [key for key in set(a) | set(b) if key not in ignored and a.get(key) != b.get(key)]
    if mismatches:
        raise ValueError('Experiments are not matched: ' + ', '.join(sorted(mismatches)))
    if a['awta'] or not b['awta']:
        raise ValueError('Expected baseline first and aWTA second')
    if baseline['split'] != modified['split'] or baseline['epoch'] != modified['epoch']:
        raise ValueError('Reports must use the same split and epoch')
    if baseline['epoch'] != a['num_epochs'] or modified['epoch'] != b['num_epochs']:
        raise ValueError('Comparison requires the predetermined final epoch')
    if baseline['groups']['all']['agents'] != modified['groups']['all']['agents']:
        raise ValueError('Report agent counts differ')
    metrics = ('minADE', 'minFDE')
    av, bv = baseline['groups']['all'], modified['groups']['all']
    if any(av[k] is None or bv[k] is None for k in metrics):
        raise ValueError('Empty evaluation split')
    result = {'split': baseline['split'], 'test_set': a['test_set'], 'seed': a['seed'],
              'baseline': {k: av[k] for k in metrics}, 'awta': {k: bv[k] for k in metrics},
              'delta_awta_minus_baseline': {k: bv[k] - av[k] for k in metrics}}
    if baseline['split'] == 'validation':
        wins = all(bv[k] <= av[k] for k in metrics) and any(bv[k] < av[k] for k in metrics)
        result['validation_selected_variant'] = 'awta' if wins else 'baseline'
        result['note'] = 'Retrain the selected method on all four training scenes. Validation selection is not a test guarantee.'
    else:
        result['note'] = 'Test reporting only. No model is selected using test scores.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('modified', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.modified.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
