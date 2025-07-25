"""
=========================================
Basic Spherical Galaxy Cluster Model
=========================================

This example constructs a simple spherical galaxy cluster model using Pisces and visualizes
its thermodynamic profiles, including temperature, entropy, and pressure.

Unlike magnetized models, this cluster assumes pure thermal pressure support. The model
uses NFW profiles for both total and gas densities and solves for the hydrostatic temperature
profile under spherical symmetry.

"""
# %%
# Setup
# -----
# We begin by importing the required profiles and model class. This example uses:
#
# - :class:`~models.galaxy_clusters.spherical.SphericalGalaxyClusterModel`
# - :class:`~profiles.density.NFWDensityProfile`

import tempfile

import matplotlib.pyplot as plt
import numpy as np
import unyt

from pisces.models.galaxy_clusters import SphericalGalaxyClusterModel
from pisces.profiles import NFWDensityProfile

# %%
# Density Profiles
# ----------------
# We'll define NFW profiles for both the total matter and gas components.

# Define the central density and scale radius for total mass.
rho_tot = unyt.unyt_quantity(5e6, "Msun/kpc**3")
r_s_tot = unyt.unyt_quantity(200, "kpc")

# Define the central density and scale radius for gas.
rho_gas = unyt.unyt_quantity(5e5, "Msun/kpc**3")
r_s_gas = unyt.unyt_quantity(220, "kpc")

# Create the profiles
total_density = NFWDensityProfile(rho_0=rho_tot, r_s=r_s_tot)
gas_density = NFWDensityProfile(rho_0=rho_gas, r_s=r_s_gas)

# %%
# Model Construction
# ------------------
# We'll construct the spherical galaxy cluster model over a logarithmic radial grid.

# Create a temporary output file
tmpdir = tempfile.TemporaryDirectory()
filename = f"{tmpdir.name}/basic_cluster_model.h5"

# Define the radial range
rmin = unyt.unyt_quantity(1.0, "kpc")
rmax = unyt.unyt_quantity(3.0, "Mpc")

# Build the model
model = SphericalGalaxyClusterModel.from_density_and_total_density(
    gas_density,
    total_density,
    filename,
    min_radius=rmin,
    max_radius=rmax,
    num_points=500,
    overwrite=True,
)

# %%
# Plot Results
# ------------
# We'll now extract and plot the thermodynamic quantities.

fields = ["temperature", "entropy", "pressure"]
labels = {
    "temperature": r"Temperature [$\mathrm{keV}$]",
    "entropy": r"Entropy [$\mathrm{keV\ cm^2}$]",
    "pressure": r"Pressure [$\mathrm{erg/cm^3}$]",
}
units = ["keV", "keV*cm**2", "erg/cm**3"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
radii = model["radii"].to("kpc").value

for ax, field, unit in zip(axes, fields, units):
    ax.plot(radii, model[field].to_value(unit), lw=2)
    ax.set_yscale("log")
    ax.set_title(labels[field])
    ax.set_xlabel("Radius [kpc]")
    ax.grid(True)

axes[0].set_ylabel("Log-scaled value")
plt.tight_layout()
plt.show()
