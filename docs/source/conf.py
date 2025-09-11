"""Sphinx configuration for the gradnet documentation site.

This config favors clear API docs from docstrings and a lightweight setup
that renders reliably on most environments.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Make package importable for autodoc (src layout), robust to CWD
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

# Pre-mock heavy optional dependencies so autosummary/autodoc can import
# our package without requiring them at build time.
def _ensure_lightweight_mocks():
    import types
    from types import ModuleType

    def _mod(name: str, pkg: bool = False) -> ModuleType:
        m = ModuleType(name)
        if pkg:
            m.__path__ = []  # mark as package for nested imports
        sys.modules[name] = m
        return m

    def _cls(name: str):
        return type(name, (), {})

    # Torch and common submodules used by our package
    try:
        import torch as _torch  # type: ignore
        _have_torch = True
    except Exception:
        _have_torch = False
    if not _have_torch:
        torch = _mod('torch', pkg=True)
        # Minimal dtypes/classes referenced in annotations/attributes
        torch.Tensor = _cls('Tensor')  # type: ignore[attr-defined]
        torch.device = _cls('device')  # type: ignore[attr-defined]
        torch.dtype = _cls('dtype')    # type: ignore[attr-defined]
        def _no_grad():
            def _decorator(fn=None):
                if fn is None:
                    return lambda f: f
                return fn
            return _decorator
        torch.no_grad = _no_grad  # type: ignore[attr-defined]
        # Submodules frequently imported
        nn = _mod('torch.nn', pkg=True)
        nn.Module = _cls('Module')     # type: ignore[attr-defined]
        nn.Parameter = _cls('Parameter')  # type: ignore[attr-defined]
        _mod('torch.nn.functional')
        tu = _mod('torch.utils', pkg=True)
        torch.utils = tu  # attach to parent
        tud = _mod('torch.utils.data', pkg=True)
        tud.Dataset = _cls('Dataset')  # type: ignore[attr-defined]
        tud.DataLoader = _cls('DataLoader')  # type: ignore[attr-defined]
        optim = _mod('torch.optim', pkg=True)
        optim.Optimizer = _cls('Optimizer')  # type: ignore[attr-defined]
        optim.Adam = _cls('Adam')  # type: ignore[attr-defined]
        torch.optim = optim  # attach to parent
        tl = _mod('torch.linalg', pkg=True)
        torch.linalg = tl  # attach to parent

    # NumPy (optional at build time)
    try:
        import numpy as _np  # noqa: F401
    except Exception:
        _mod('numpy')

    # NetworkX
    try:
        import networkx as _nx  # noqa: F401
    except Exception:
        _mod('networkx')

    # PyTorch Lightning and submodules referenced in docs
    try:
        import pytorch_lightning as _pl  # noqa: F401
        _have_pl = True
    except Exception:
        _have_pl = False
    if not _have_pl:
        pl = _mod('pytorch_lightning', pkg=True)
        pl.LightningModule = _cls('LightningModule')  # minimal base class
        def _seed_everything(seed=None, workers=None):
            return None
        pl.seed_everything = _seed_everything  # type: ignore[attr-defined]
        # callbacks
        cbmod = _mod('pytorch_lightning.callbacks', pkg=True)
        cbmod.Callback = _cls('Callback')  # type: ignore[attr-defined]
        cbmod.ModelCheckpoint = _cls('ModelCheckpoint')  # type: ignore[attr-defined]
        # loggers
        lg = _mod('pytorch_lightning.loggers', pkg=True)
        lgl = _mod('pytorch_lightning.loggers.logger')
        lgl.Logger = _cls('Logger')  # type: ignore[attr-defined]
        # utilities.warnings
        utilw = _mod('pytorch_lightning.utilities', pkg=True)
        utilw_warn = _mod('pytorch_lightning.utilities.warnings')
        class _PLPossibleUserWarning(Warning):
            pass
        utilw_warn.PossibleUserWarning = _PLPossibleUserWarning  # type: ignore[attr-defined]

    # tqdm (only for type/attribute presence)
    try:
        import tqdm.auto as _tqa  # noqa: F401
    except Exception:
        _mod('tqdm', pkg=True)
        tqa = _mod('tqdm.auto')
        def _tqdm(*args, **kwargs):
            return args[0] if args else None
        tqa.tqdm = _tqdm  # type: ignore[attr-defined]

_ensure_lightweight_mocks()

# Make type hint resolution resilient during docs build
try:
    import typing as _typing
    _orig_get_type_hints = _typing.get_type_hints  # type: ignore[attr-defined]
    def _safe_get_type_hints(obj, *args, **kwargs):  # type: ignore[override]
        try:
            return _orig_get_type_hints(obj, *args, **kwargs)
        except Exception:
            return {}
    _typing.get_type_hints = _safe_get_type_hints  # type: ignore[assignment]
except Exception:
    pass

# -- Project information -----------------------------------------------------
project = 'gradnet'
author = 'Guram Mikaberidze, Beso Mikaberidze, Dane Taylor'
copyright = '2025, ' + author
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
autodoc_member_order = 'bysource'
autoclass_content = 'both'           # include class and __init__ docstrings
autodoc_typehints = 'none'           # remove type hints from signatures
autodoc_default_options = {
    'exclude-members': '__init__',
}
autodoc_mock_imports = []

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Remove legacy typehints overrides (unused without extension)
try:
    del typehints_use_signature  # type: ignore[name-defined]
    del typehints_use_signature_return  # type: ignore[name-defined]
except Exception:
    pass

# Markdown / Notebooks (MyST-NB)
# Enable linkify only if the optional dependency is installed.
try:
    import linkify_it  # noqa: F401
    _myst_ext = ['colon_fence', 'deflist', 'dollarmath', 'amsmath', 'linkify']
except Exception:
    _myst_ext = ['colon_fence', 'deflist', 'dollarmath', 'amsmath']
myst_enable_extensions = _myst_ext
# Allow `$$ ... $$` used inline; block-display works regardless
myst_dmath_double_inline = True
# Render notebooks inline without executing during build
nb_execution_mode = 'off'
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# Prefer RTD theme if available; fall back otherwise.
try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = 'sphinx_rtd_theme'
except Exception:
    html_theme = 'alabaster'

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# Cross-links to external projects
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# --- Auto-link example notebooks into docs/source/tutorials -----------------
def _link_example_notebooks():
    """Sync example notebooks into ``source/tutorials`` for the Tutorials toc.

    - Only include notebooks that follow the "<number>_<name>.ipynb" pattern
      (e.g., ``1_mean_shortest_path.ipynb``, ``2_kuramoto.ipynb``).
    - Create symlinks (or copies if symlinks are not permitted) inside
      ``docs/source/tutorials`` so the ``:glob:`` toctree picks them up.
    - Remove stale links/files in ``tutorials/`` that do not match the pattern
      or no longer exist at the source.
    """
    import re
    import shutil

    src_dir = Path(__file__).parent.parent.parent / 'examples'
    dst_dir = Path(__file__).parent / 'tutorials'

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    pattern = re.compile(r"^\d+_.+\.ipynb$")

    # Collect desired source notebooks
    desired = {nb.name: nb for nb in src_dir.glob('*.ipynb') if pattern.match(nb.name)}

    # Clean up stale or non-conforming files in destination
    for item in dst_dir.glob('*.ipynb'):
        try:
            if item.name not in desired:
                # Remove any notebook not matching pattern or no longer present
                item.unlink()
                continue
            if item.is_symlink():
                # Drop broken symlinks
                try:
                    _ = item.resolve(strict=True)
                except FileNotFoundError:
                    item.unlink()
        except Exception:
            # Ignore cleanup errors; continue syncing
            pass

    # Create/update links for desired notebooks
    for name, nb in desired.items():
        target = dst_dir / name
        try:
            if target.exists():
                # Already present and valid
                continue
            # Prefer symlink; fall back to copy if not permitted
            try:
                target.symlink_to(nb.resolve())
            except Exception:
                shutil.copy2(nb, target)
        except Exception:
            # Best-effort; keep building even if a single file fails
            pass

def setup(app):  # noqa: D401
    """Sphinx hook to prepare docs before build."""
    _link_example_notebooks()
    # Disable autodoc typehints handler which can choke on mocked types
    def _disconnect_typehints(_app):
        try:
            from sphinx.ext.autodoc.typehints import record_typehints
            _app.disconnect('autodoc-process-signature', record_typehints)
        except Exception:
            pass
    app.connect('builder-inited', _disconnect_typehints)
