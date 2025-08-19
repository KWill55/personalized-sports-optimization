import yaml
from pathlib import Path

def load_config(config_path: str | Path) -> dict:
    """Load the YAML config as a dictionary."""
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_paths(config_path: str | Path, athlete: str = None, session: str = None) -> dict:
    """
    Load paths from project_config.yaml and substitute {athlete} and {session}.
    Returns a dict of Path objects.
    """
    cfg = load_config(config_path)

    # Use overrides if provided, otherwise fall back to YAML values
    athlete = athlete or cfg.get("athlete")
    session = session or cfg.get("session")

    if not athlete or not session:
        raise ValueError("Both athlete and session must be defined (either in YAML or arguments).")

    paths_cfg = cfg.get("paths", {})
    resolved = {}

    for key, raw_path in paths_cfg.items():
        # Substitute placeholders
        path_str = raw_path.format(athlete=athlete, session=session)
        resolved[key] = Path(path_str)

    return resolved
