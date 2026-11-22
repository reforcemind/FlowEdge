import unittest
from unittest.mock import patch

from flowedge_lerobot.__main__ import main


class EntrypointTests(unittest.TestCase):
    def test_dispatch_forwards_arguments_and_status(self):
        with patch("flowedge_lerobot.deployment.main", return_value=7) as package:
            self.assertEqual(main(["package", "source", "output"]), 7)
        package.assert_called_once_with(["source", "output"])

    def test_command_help_is_forwarded(self):
        with patch("flowedge_lerobot.preflight.main", return_value=0) as preflight:
            self.assertEqual(main(["preflight", "--help"]), 0)
        preflight.assert_called_once_with(["--help"])

    def test_unknown_command_is_rejected(self):
        with self.assertRaises(SystemExit) as error:
            main(["unknown"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
