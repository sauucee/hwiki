"""The optional QMD backend must not block the default FTS provider."""
from pathlib import Path


def test_qmd_is_not_declared_as_a_required_external_dependency():
    import yaml

    manifest = yaml.safe_load((Path(__file__).resolve().parents[1] / 'plugin.yaml').read_text())
    required = manifest.get('external_dependencies') or []
    assert all(dep.get('name') != 'qmd' for dep in required)
