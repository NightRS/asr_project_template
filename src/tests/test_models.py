import unittest

import torch

from .utils import models_to_test_instances


class TestTrainer(unittest.TestCase):
    @torch.no_grad()
    def test_shape(self):
        batch_size, n_mels, max_time = 9, 64, 101
        alphabet_size = 28

        for model in models_to_test_instances(n_mels, alphabet_size):
            with self.subTest(model=type(model).__name__):
                inputs = torch.randn(
                    batch_size,
                    n_mels,
                    max_time,
                    generator=torch.Generator().manual_seed(0),
                )
                outputs = model(spectrogram=inputs)["logits"]
                self.assertEqual(
                    outputs.shape,
                    torch.Size(
                        (
                            batch_size,
                            model.transform_input_lengths(max_time),
                            alphabet_size,
                        )
                    ),
                )

    def test_inputs_independence(self):
        batch_size, n_mels, max_time = 9, 64, 101
        alphabet_size = 28

        for model in models_to_test_instances(n_mels, alphabet_size):
            with self.subTest(model=str(model)):
                for mask_idx in range(batch_size):
                    inputs = torch.randn(
                        batch_size,
                        n_mels,
                        max_time,
                        generator=torch.Generator().manual_seed(0),
                        requires_grad=True,
                    )

                    model.zero_grad()
                    model.eval()
                    outputs = model(inputs)["logits"]
                    model.train()

                    mask = torch.ones_like(outputs)
                    mask[mask_idx] = 0
                    outputs = outputs * mask

                    loss = outputs.mean()
                    loss.backward()

                    assert inputs.grad is not None
                    for i, grad in enumerate(inputs.grad):
                        if i == mask_idx:
                            self.assertTrue(torch.all(grad == 0).item())
                        else:
                            self.assertFalse(torch.all(grad == 0).item())
