from types import SimpleNamespace

import pytest

from baf.runtime import resolve_device
from baf.runtime import environment_report


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_device_priority(cuda, mps, expected):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda), backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))
    assert resolve_device("auto", torch) == expected
    assert resolve_device("cpu", torch) == "cpu"
    if not cuda:
        with pytest.raises(ValueError, match="unavailable"):
            resolve_device("cuda", torch)


def test_broken_torch_dependency_is_not_reported_as_uninstalled(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("missing internal dependency", name="torch_dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(ModuleNotFoundError, match="internal dependency"):
        environment_report()
