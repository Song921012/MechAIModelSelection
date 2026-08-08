from pathlib import Path
import unittest

from mechai_experiments.run import load_profile, command


class LauncherTests(unittest.TestCase):
    def test_smoke_is_the_default_small_profile(self):
        profile = load_profile("smoke")
        self.assertEqual(profile["profile"], "smoke")
        self.assertEqual(profile["seeds"]["core"], 1)
        self.assertEqual(profile["seeds"]["cross-domain"], 1)

    def test_submission_command_is_explicit(self):
        profile = load_profile("submission")
        cmd = command("core", "submission", 3, True, profile)
        self.assertIn("submission", cmd)
        self.assertIn("--resume", cmd)
        self.assertEqual(cmd[cmd.index("--workers") + 1], "3")

    def test_reference_uses_released_records(self):
        profile = load_profile("smoke")
        cmd = command("reference", "smoke", 1, True, profile)
        self.assertEqual(cmd[-2:], ["--source", "submission"])


if __name__ == "__main__":
    unittest.main()
