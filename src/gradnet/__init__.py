# src/gradnet/__init__.py
import importlib
from typing import TYPE_CHECKING

__all__ = [
    "GradNet",
    "integrate_ode",
    "fit",
]

_LAZY_ATTRS = {
    "GradNet": (".gradnet", "GradNet"),
    "integrate_ode": (".ode", "integrate_ode"),
    "fit": (".trainer", "fit"),
}

def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as e:
        raise AttributeError(f"module 'gradnet' has no attribute {name!r}") from e
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    # Keep the original __module__ to avoid confusing autodoc source analysis
    globals()[name] = value  # cache for future access
    return value

def __dir__():
    return sorted(list(globals().keys()) + __all__)

if TYPE_CHECKING:
    # For type checkers, keep explicit imports without runtime cost
    from .gradnet import GradNet  # noqa: F401
    from .ode import integrate_ode  # noqa: F401
    from .trainer import fit  # noqa: F401
