import unittest

from tqdm import tqdm

from src.collate_fn.collate import collate_fn
from src.datasets import LibrispeechDataset
from src.tests.utils import clear_log_folder_after_use
from src.utils.object_loading import get_dataloader_from_params, get_datasets
from src.utils.parse_config import ConfigParser


class TestDataloader(unittest.TestCase):
    def test_collate_fn(self):
        config_parser = ConfigParser.get_test_configs(run_name="test_collate_fn")
        with clear_log_folder_after_use(config_parser):
            ds = LibrispeechDataset(
                "dev-clean",
                text_encoder=config_parser.get_text_encoder(),
                config_parser=config_parser,
            )

            batch_size = 3
            batch = collate_fn([ds[i] for i in range(batch_size)])

            self.assertIn("spectrogram", batch)  # torch.tensor
            batch_size_dim, feature_length_dim, time_dim = batch["spectrogram"].shape
            self.assertEqual(batch_size_dim, batch_size)
            self.assertEqual(feature_length_dim, 64)

            self.assertIn("text_encoded", batch)  # [int] torch.tensor
            # joined and padded indexes representation of transcriptions
            batch_size_dim, text_length_dim = batch["text_encoded"].shape
            self.assertEqual(batch_size_dim, batch_size)

            self.assertIn("text_encoded_length", batch)  # [int] torch.tensor
            # contains lengths of each text entry
            self.assertEqual(len(batch["text_encoded_length"].shape), 1)
            batch_size_dim = batch["text_encoded_length"].shape[0]
            self.assertEqual(batch_size_dim, batch_size)

            self.assertIn("text", batch)  # list[str]
            # simple list of initial normalized texts
            batch_size_dim = len(batch["text"])
            self.assertEqual(batch_size_dim, batch_size)

    def test_dataloaders(self):
        _TOTAL_ITERATIONS = 10
        config_parser = ConfigParser.get_test_configs(run_name="test_dataloaders")
        with clear_log_folder_after_use(config_parser):
            dataset_and_params = get_datasets(
                config_parser, config_parser.get_text_encoder()
            )
            for part in ["train", "val"]:
                dl = get_dataloader_from_params(*dataset_and_params[part])
                for i, batch in tqdm(
                    enumerate(iter(dl)),
                    total=_TOTAL_ITERATIONS,
                    desc=f"Iterating over {part}",
                ):
                    if i >= _TOTAL_ITERATIONS:
                        break
