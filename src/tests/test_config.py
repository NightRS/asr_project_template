import json
import unittest

from src.tests.utils import clear_log_folder_after_use
from src.utils.parse_config import ConfigParser


class TestConfig(unittest.TestCase):
    def test_create(self):
        config_parser = ConfigParser.get_test_configs(run_name="test_config_create")
        with clear_log_folder_after_use(config_parser):
            json.dumps(config_parser.config, indent=2)
