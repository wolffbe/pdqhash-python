from .bindings import compute, compute_dihedral, compute_float

try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("pdqhash")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["compute", "compute_dihedral", "compute_float", "__version__"]
