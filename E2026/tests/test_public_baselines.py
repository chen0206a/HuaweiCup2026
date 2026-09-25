"""Public-baseline adapters must respect the audited time mask."""
import unittest

import torch

from src.models.public_baselines import MODELS


class PublicBaselineInterfaceTest(unittest.TestCase):
    def test_padding_is_inert_and_heads_have_common_shape(self):
        torch.manual_seed(7)
        valid = torch.arange(50)[None, :] < torch.tensor([1, 17])[:, None]
        batch = {"padding_mask": valid,
                 "text": torch.randn(2, 50, 768),
                 "audio": torch.randn(2, 50, 74),
                 "vision": torch.randn(2, 50, 35)}
        changed = {key: value.clone() for key, value in batch.items()}
        for modality in ("text", "audio", "vision"):
            changed[modality][~valid] = 1000 * torch.randn_like(changed[modality][~valid])
        for name, cls in MODELS.items():
            with self.subTest(model=name):
                model = cls().eval()
                original = model(batch)
                padded_changed = model(changed)
                self.assertEqual(tuple(original["classification_logits"].shape), (2, 3))
                self.assertEqual(tuple(original["regression"].shape), (2,))
                for key in original:
                    torch.testing.assert_close(original[key], padded_changed[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
