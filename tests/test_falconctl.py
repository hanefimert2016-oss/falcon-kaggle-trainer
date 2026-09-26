import importlib.util
import json
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("falconctl", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_render_metadata(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    mod = load_module(root / "falconctl.py")
    monkeypatch.setenv("KAGGLE_OWNER", "demo-user")
    monkeypatch.setattr(mod, "KAGGLE_DIR", tmp_path)
    monkeypatch.setattr(mod, "META_PATH", tmp_path / "kernel-metadata.json")
    (tmp_path / "train_flm.py").write_text("print('ok')\n", encoding="utf-8")
    p = mod.render("falcon-flm-train", "NvidiaTeslaT4")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["id"] == "demo-user/falcon-flm-train"
    assert data["enable_gpu"] is True
    assert data["machine_shape"] == "NvidiaTeslaT4"
    assert data["code_file"] == "train_flm.py"


def test_token_auth_is_accepted(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    mod = load_module(root / "falconctl.py")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fake-token-for-unit-test")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    mod._require_auth()
