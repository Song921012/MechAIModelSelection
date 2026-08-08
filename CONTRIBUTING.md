# Contributing

Please open an issue before changing a protocol or record schema. Small fixes
should include a smoke test and must preserve existing JSON compatibility.
Never overwrite submission records without an explicit protocol change.

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m mechai_experiments.analyze --audit
```
