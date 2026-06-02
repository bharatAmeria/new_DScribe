from pathlib import Path
import yaml


def _find_root() -> Path:
    """Walk up from this file until config.yaml is found (project root)."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "config.yaml").exists():
            return parent
    return Path.cwd()


class _Box(dict):
    """Simple dot-access dict wrapper (like python-box, no extra dep)."""
    def __getattr__(self, key: str):
        try:
            val = self[key]
            return _Box(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: Path | None = None) -> _Box:
    if path is None:
        path = _find_root() / "config.yaml"
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _Box(raw)


CONFIG = load_config()
