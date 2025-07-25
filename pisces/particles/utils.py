"""Particle dataset utilities."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ParticleDataset


def concatenate_particles(
    output_path: str | Path,
    *datasets: "ParticleDataset",
    overwrite: bool = False,
    groups: list[str] = None,
    **kwargs,
) -> "ParticleDataset":
    """Concatenate multiple :class:`ParticleDataset` objects into a new dataset file.

    This method creates a new dataset by copying the first input dataset to `output_path`
    and then concatenates the remaining datasets into it using the `concatenate_inplace` method.

    Parameters
    ----------
    output_path : str or Path
        Path where the new HDF5 file will be created.
    *datasets : ParticleDataset
        A sequence of particle datasets to concatenate. Must include at least one dataset.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists. Defaults to False.
    groups : list of str, optional
        Specific groups to concatenate. If None, all groups in each dataset will be used.
    **kwargs
        Additional keyword arguments passed to the final dataset constructor.

    Returns
    -------
    ParticleDataset
        A new dataset containing the concatenated data.

    Raises
    ------
    ValueError
        If no input datasets are provided.
    FileExistsError
        If the output path exists and `overwrite` is False.

    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset must be provided for concatenation.")

    # Copy the first dataset to the new file
    base = datasets[0].copy(output_path, overwrite=overwrite, **kwargs)

    # Concatenate the rest into it
    base.concatenate_inplace(*datasets[1:], groups=groups)

    return base
