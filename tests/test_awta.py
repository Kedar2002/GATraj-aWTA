"""CPU regression, gradient, prediction and exact-resume checks."""
import contextlib
import io
import random
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import torch
from awta import temperature, assignment_weights, annealed_regression
from models import GATraj
from Processor import Processor
from train import get_parser, prepare_seed, experiment_name
from compare import compare


def args(**changes):
    result = get_parser().parse_args([])
    for name, value in changes.items():
        setattr(result, name, value)
    return result


def inputs():
    absolute = torch.arange(20 * 3 * 2, dtype=torch.float32).reshape(20, 3, 2) / 100
    normalized = absolute - absolute[7:8]
    mask = np.ones((3, 3)) - np.eye(3)
    return (absolute, normalized, [np.repeat(mask[None], 20, axis=0)],
            torch.full((3, 20), 2.), np.array([[0, 3]]))


def model(a):
    with contextlib.redirect_stdout(io.StringIO()):
        return GATraj(a)


class LossTests(unittest.TestCase):
    def test_absolute_epoch_schedule(self):
        self.assertEqual(temperature(1, 1000), .5)
        self.assertAlmostEqual(temperature(300, 1000), .01)
        self.assertEqual(temperature(301, 1000), 0)
        self.assertGreater(temperature(200, 1000), temperature(250, 1000))

    def test_gradients_and_hard_limit(self):
        loc = torch.tensor([.1, .4, .8]).reshape(3, 1, 1, 1).expand(-1, 2, 12, 2).clone().requires_grad_()
        scale = torch.ones_like(loc)
        target = torch.zeros(2, 12, 2)
        ade = loc.norm(dim=-1).mean(-1)
        weights = assignment_weights(ade, .5)
        self.assertFalse(weights.requires_grad)
        torch.testing.assert_close(weights.sum(0), torch.ones(2))
        loss = annealed_regression(torch.cat((loc, scale), -1), target, ade, .5)
        loss.backward()
        self.assertTrue((loc.grad.abs().sum((1, 2, 3)) > 0).all())
        torch.testing.assert_close(assignment_weights(ade, 1e-6), assignment_weights(ade, 0))
        hard = annealed_regression(torch.cat((loc, scale), -1), target, ade, 0)
        self.assertAlmostEqual(hard.item(), np.log(2) + .1, places=6)

    def test_baseline_formula_and_prediction_equivalence(self):
        prepare_seed(4)
        baseline = model(args(awta=False)).eval()
        prepare_seed(4)
        modified = model(args(awta=True)).eval()
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, modified.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(list(baseline.state_dict()), list(modified.state_dict()))
        x = inputs()
        with torch.no_grad():
            mixture = baseline(x, 1, return_mixture=True)
            other = modified(x, 1, return_mixture=True)
            for a, b in zip(mixture, other):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            mu, sigma, pi = mixture
            target = x[1][8:].transpose(0, 1)
            distance = (mu - target).norm(dim=-1).sum(-1)
            winner = distance.argmin(0)
            original_reg = baseline.reg_loss(torch.cat((mu, sigma), -1)[winner, torch.arange(3)], target)
            original_cls = baseline.cls_loss(pi, torch.softmax(-distance / 12, 0).t())
            loss, _ = baseline(x, 1)
            torch.testing.assert_close(loss, original_reg + original_cls, rtol=0, atol=0)
            changed = list(x)
            changed[0] = x[0].clone(); changed[1] = x[1].clone()
            changed[0][8:] += 100; changed[1][8:] -= 100
            for a, b in zip(mixture, baseline(tuple(changed), 1, return_mixture=True)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        modified.train()
        modified(x, 1)[0].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in modified.parameters() if p.grad is not None))

    def test_gaussian_hard_limit_and_small_scales(self):
        loc = torch.tensor([.1, .5]).reshape(2, 1, 1, 1).expand(-1, 2, 12, 2)
        scale = torch.ones_like(loc)
        target = torch.zeros(2, 12, 2)
        ade = loc.norm(dim=-1).mean(-1)
        hard = annealed_regression(torch.cat((loc, scale), -1), target, ade, 0, gaussian=True)
        self.assertAlmostEqual(hard.item(), .5 * .1**2, places=7)
        for gaussian in (False, True):
            value = annealed_regression(torch.cat((loc, scale * 1e-8), -1), target, ade, .5, gaussian=gaussian)
            self.assertTrue(torch.isfinite(value))

    def test_names_separate_variants_and_schedules(self):
        self.assertNotEqual(experiment_name(args(awta=False)), experiment_name(args(awta=True)))
        self.assertNotEqual(experiment_name(args(awta=True)), experiment_name(args(awta=True, num_epochs=500)))
        self.assertNotEqual(experiment_name(args()), experiment_name(args(validation_set=4)))


class ResumeTests(unittest.TestCase):
    def processor(self, directory, load=0, **changes):
        prepare_seed(8)
        p = Processor.__new__(Processor)
        p.args = args(save_dir=directory, model_dir=str(Path(directory) / 'check'),
                      train_model='check', load_model=load, awta=True, **changes)
        p.net = torch.nn.Linear(2, 2)
        p.lr = p.args.learning_rate
        p.set_optimizer()
        p.epoch = 1
        p.load_model()
        return p

    def step(self, p, epoch):
        p.optimizer.zero_grad()
        noise = torch.randn(3, 2) + random.random() + np.random.random()
        # Exercise absolute-epoch temperature and RNG restoration in one step.
        loss = p.net(noise).square().mean() * (1 + temperature(epoch, p.args.num_epochs))
        loss.backward(); p.optimizer.step(); p.scheduler.step()

    def test_exact_resume_zip_and_configuration_guard(self):
        (Path(__file__).resolve().parents[1] / 'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'outputs') as directory:
            (Path(directory) / 'check').mkdir()
            full = self.processor(directory)
            for epoch in range(1, 5): self.step(full, epoch)
            segmented = self.processor(directory)
            for epoch in range(1, 3): self.step(segmented, epoch)
            segmented.save_model(2)
            checkpoint_path = Path(directory) / 'check' / 'check_2.tar'
            self.assertTrue(zipfile.is_zipfile(checkpoint_path))
            self.assertFalse(Path(str(checkpoint_path) + '.tmp').exists())
            resumed = self.processor(directory, load=2)
            self.assertEqual(resumed.epoch, 3)
            for epoch in range(3, 5): self.step(resumed, epoch)
            for key, value in full.net.state_dict().items():
                torch.testing.assert_close(value, resumed.net.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(full.scheduler.state_dict(), resumed.scheduler.state_dict())
            with self.assertRaisesRegex(ValueError, 'awta_initial'):
                self.processor(directory, load=2, awta_initial=.8)


class SelectionTests(unittest.TestCase):
    def report(self, modified, split, ade, fde):
        return {'experiment': {'awta': modified, 'num_epochs': 1000, 'test_set': 0, 'seed': 1},
                'split': split, 'epoch': 1000,
                'groups': {'all': {'agents': 10, 'minADE': ade, 'minFDE': fde}}}

    def test_validation_fallback_and_no_test_selection(self):
        a = self.report(False, 'validation', .2, .4)
        worse = self.report(True, 'validation', .19, .41)
        better = self.report(True, 'validation', .19, .39)
        self.assertEqual(compare(a, worse)['validation_selected_variant'], 'baseline')
        self.assertEqual(compare(a, better)['validation_selected_variant'], 'awta')
        a['split'] = better['split'] = 'test'
        self.assertNotIn('validation_selected_variant', compare(a, better))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
