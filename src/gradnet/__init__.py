# src/gradnet/__init__.py
import importlib
from typing import TYPE_CHECKING

__all__ = [
    "GradNet",
    "integrate_ode",
    "fit",
    "pl_fit",
]

_LAZY_ATTRS = {
    "GradNet": (".gradnet", "GradNet"),
    "integrate_ode": (".ode", "integrate_ode"),
    "fit": (".pl_trainer", "fit"),         # will move to .trainer once the custom trainer lands
    "pl_fit": (".pl_trainer", "fit"),
}

def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as e:
        raise AttributeError(f"module 'gradnet' has no attribute {name!r}") from e
    try:
        module = importlib.import_module(module_name, __name__)
    except ImportError as e:
        if "pytorch_lightning" in str(e):
            raise ImportError(
                f"gradnet.{name} requires pytorch-lightning. "
                f"Install with: pip install 'gradnet[pl]'"
            ) from e
        raise
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

def __dir__():
    return sorted(list(globals().keys()) + __all__)

if TYPE_CHECKING:
    from .gradnet import GradNet  # noqa: F401
    from .ode import integrate_ode  # noqa: F401
    from .pl_trainer import fit  # noqa: F401
    from .pl_trainer import fit as pl_fit  # noqa: F401
