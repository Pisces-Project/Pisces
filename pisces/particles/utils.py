"""Particle dataset utilities."""

from pathlib import Path
from typing import TYPE_CHECKING, Union

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


def inspect_particle_count(path: Union[str, Path]) -> dict[str, int]:
    """
    Inspect the number of particles in each species group of a particle dataset without fully loading the file.

    This function:

      1. Opens the HDF5 particle file at the top level.
      2. Iterates over all top-level groups (each representing a particle species).
      3. Skips any group with the attribute ``NOT_PARTICLE_GROUP`` set to a truthy value.
      4. Reads the ``NUMBER_OF_PARTICLES`` attribute from each valid particle group.

    Parameters
    ----------
    path : str or ~pathlib.Path
        Path to the HDF5 particle file.

    Returns
    -------
    dict
        Mapping of particle group names to integer particle counts.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is not a regular file.
    ValueError
        If a valid particle group is missing the required ``NUMBER_OF_PARTICLES`` attribute.
    """
    import h5py

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No particle file found at: {p}")

    counts: dict[str, int] = {}
    with h5py.File(p, "r") as f:
        for group_name, group in f.items():
            if not isinstance(group, h5py.Group):
                continue
            if bool(group.attrs.get("NOT_PARTICLE_GROUP", False)):
                continue
            if "NUMBER_OF_PARTICLES" not in group.attrs:
                raise ValueError(
                    f"Particle group '{group_name}' in '{p}' is missing the 'NUMBER_OF_PARTICLES' attribute."
                )
            counts[group_name] = int(group.attrs["NUMBER_OF_PARTICLES"])

    return counts


def inspect_species(path: Union[str, Path]) -> list[str]:
    """
    List the particle species (top-level groups) present in a particle dataset.

    This function opens the given HDF5 particle file and inspects its immediate
    top-level groups. Any group with the attribute ``NOT_PARTICLE_GROUP`` set
    to a truthy value will be skipped. All other top-level groups are assumed
    to represent particle species (e.g., ``"PartType0"``, ``"PartType1"``, etc.).

    Parameters
    ----------
    path : str or ~pathlib.Path
        Path to the HDF5 particle file.

    Returns
    -------
    list of str
        Names of particle species present in the dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is not a regular file.
    """
    import h5py

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No particle file found at: {p}")

    species: list[str] = []
    with h5py.File(p, "r") as f:
        for name, obj in f.items():
            if not isinstance(obj, h5py.Group):
                continue
            if bool(obj.attrs.get("NOT_PARTICLE_GROUP", False)):
                continue
            species.append(name)

    return species


def inspect_fields(path: Union[str, Path]) -> dict[str, list[tuple[str, tuple[int, ...]]]]:
    """
    Inspect the available fields for each particle species in a dataset without fully loading it.

    This function opens the HDF5 particle file, iterates over its top-level
    groups (each representing a particle species), skips any groups marked
    with the ``NOT_PARTICLE_GROUP`` attribute set to a truthy value, and
    collects the names and *per-particle shapes* of all datasets in each group.

    The "per-particle shape" is defined as ``dataset.shape[1:]``—the shape of
    each individual particle's data entry (excluding the leading dimension
    which counts particles).

    Parameters
    ----------
    path : str or ~pathlib.Path
        Path to the HDF5 particle file.

    Returns
    -------
    dict[str, list[tuple[str, tuple[int, ...]]]]
        Mapping of particle species names to a list of ``(field_name, element_shape)``
        tuples, where ``element_shape`` is the shape of one particle's data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is not a regular file.
    """
    import h5py

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No particle file found at: {p}")

    fields: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
    with h5py.File(p, "r") as f:
        for group_name, group in f.items():
            if not isinstance(group, h5py.Group):
                continue
            if bool(group.attrs.get("NOT_PARTICLE_GROUP", False)):
                continue

            group_fields = []
            for field_name, ds in group.items():
                if isinstance(ds, h5py.Dataset):
                    element_shape = ds.shape[1:]  # shape per particle
                    group_fields.append((field_name, element_shape))

            fields[group_name] = group_fields

    return fields
