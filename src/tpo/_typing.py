"""Internal typing helpers shared across JAX-heavy modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import Any, TypeAlias, cast

import jax
import numpy as np

Array: TypeAlias = jax.Array
PRNGKey: TypeAlias = jax.Array
PyTree: TypeAlias = Any
Params: TypeAlias = Any
OptState: TypeAlias = Any
PathLikeStr: TypeAlias = str | PathLike[str]
MetricSeries: TypeAlias = Sequence[float] | np.ndarray[Any, Any] | Array
MetricMap: TypeAlias = Mapping[str, MetricSeries]
NumpyArray: TypeAlias = np.ndarray[Any, Any]
ArrayMap: TypeAlias = Mapping[str, np.ndarray[Any, Any]]
AlgoResults: TypeAlias = dict[str, NumpyArray]
ResultsByLength: TypeAlias = dict[int, AlgoResults]
ScopedResultsByLength: TypeAlias = dict[str, ResultsByLength]
VariantKey: TypeAlias = tuple[str, str]
VariantResults: TypeAlias = dict[VariantKey, AlgoResults]
ScopedVariantResults: TypeAlias = dict[str, VariantResults]
SeedRunner: TypeAlias = Callable[[Array], Array]


def savez_dict(path: PathLikeStr, arrays: Mapping[str, object]) -> None:
    """Persist a mapping of named arrays with a type-checker-friendly wrapper."""

    cast(Any, np.savez)(path, **dict(arrays))
