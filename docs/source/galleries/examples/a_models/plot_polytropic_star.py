"""
=============================
Build a Polytropic Star Model
=============================

This example shows how to use the :class:`~pisces.models.stars.polytropes.PolytropicStarModel`
to generate the internal structure of a star governed by a polytropic equation of state.

We demonstrate two ways to construct a model:

1. From the total **mass and radius** of the star.
2. From the **central density and central temperature** of the core.

The resulting model outputs physical profiles for density, pressure, temperature, gravitational
potential, and mass.
"""

import tempfile

import matplotlib.pyplot as plt

# %%
# Imports
# -------
import numpy as np
from unyt import unyt_quantity

from pisces.models.stars.polytropes import PolytropicStarModel

# Create a temp directory.
tmpdir = tempfile.TemporaryDirectory()

# %%
# Generate a Polytropic Star from Mass and Radius
# -----------------------------------------------
# Here we create a model for a star with 1 solar mass and 1 solar radius,
# assuming a polytropic index n = 1.5 (typical for fully convective stars).
mass = unyt_quantity(1, "Msun")
radius = unyt_quantity(1, "Rsun")
n_index = [0.2, 0.3, 0.5, 1, 1.25, 1.5, 2, 2.5, 3]

models = []

for n in n_index:
    filename = f"{tmpdir.name}/polytrope_example_{n}.h5"

    models.append(
        PolytropicStarModel.from_mass_and_radius(
            filename=filename,
            mass=mass,
            radius=radius,
            polytropic_index=n,
            overwrite=True,
            rmax=unyt_quantity(10, "Rsun"),
            rmin=unyt_quantity(0.01, "Rsun"),
        )
    )

# %%
# Visualize the Stellar Structure
# -------------------------------
# We now plot four key physical fields as a function of radius:
# density, temperature, pressure, and gravitational field.
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
cmap = plt.cm.cool
norm = plt.Normalize(vmin=min(n_index), vmax=max(n_index))

for model, n in zip(models, n_index):
    # Scale n down to 0-1 for color.
    color = cmap(norm(n))

    # Add the plots.
    axs[0, 0].plot(model.grid["r"].to("Rsun").value, model["density"].to("g/cm**3").value, color=color)
    axs[0, 1].plot(model.grid["r"].to("Rsun").value, model["temperature"].to("K").value, color=color)
    axs[1, 0].plot(model.grid["r"].to("Rsun").value, model["mass"].to("Msun").value, color=color)
    axs[1, 1].plot(model.grid["r"].to("Rsun").value, model["gravitational_field"].to("cm/s**2").value, color=color)

# Scales
for ax in axs.ravel():
    ax.set_xscale("log")
    ax.set_yscale("log")


# Axis labels
axs[0, 0].set_ylabel(r"Density [${\rm g\;cm^{-3}}$]")
axs[0, 1].set_ylabel(r"Temperature [${\rm K}$]")
axs[1, 0].set_ylabel(r"Enclosed Mass [${\rm M_\odot}$]")
axs[1, 1].set_ylabel(r"Gravitational Field [${\rm cm\;s^{-2}}$]")
for ax in axs[1]:
    ax.set_xlabel(r"Radius [${\rm R_\odot}$]")

# Titles
axs[0, 0].set_title("Density Profile")
axs[0, 1].set_title("Temperature Profile")
axs[1, 0].set_title("Mass Profile")
axs[1, 1].set_title("Gravitational Field")

# Add colorbar for polytropic index
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.025, pad=0.02)
cbar.set_label("Polytropic Index (n)")

plt.show()
