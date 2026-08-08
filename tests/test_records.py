import json
from pathlib import Path
import tempfile
import unittest

from mechai_experiments.records import SCHEMA_VERSION, load_compatible, protocol_hash, result_path, write_record


class RecordTests(unittest.TestCase):
    def test_result_path_uses_submission_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = result_path(Path(tmp), "core", "example")
            self.assertIn(str(Path("results") / "records" / "submission" / "core"), str(path))

    def test_round_trip_and_protocol_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = result_path(root, "core", "example")
            digest = protocol_hash({"seed": 1})
            write_record(path, digest, {"status": "ok", "value": 2.0})
            record = load_compatible(path, digest, False)
            self.assertEqual(record["schema_version"], SCHEMA_VERSION)
            self.assertEqual(record["value"], 2.0)
            with self.assertRaises(RuntimeError):
                load_compatible(path, "different", False)


if __name__ == "__main__":
    unittest.main()
