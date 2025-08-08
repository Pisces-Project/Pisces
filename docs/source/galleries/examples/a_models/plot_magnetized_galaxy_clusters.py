"""
====================================
Comparing Magnetized Galaxy Clusters
====================================

This example generates a set of magnetized galaxy cluster models with varying
magnetic pressure fractions (beta parameters) and compares their resulting
temperature, entropy, magnetic field strength, and Alfven speed profiles.

Each model uses the same density profile but varies the magnetic support,
demonstrating the effects of magnetization on thermodynamic quantities.

"""
# %%
# Setup
# -----
# In this example, we'll use the :class:`~models.galaxy_clusters.spherical.MagnetizedSphericalGalaxyClusterModel` class
# to construct magnetized galaxy clusters with different degrees of physical support. This is
# parameterized through the :math:`\beta` parameter which is defined such that
#
# .. math::
#
#   \beta = \frac{P_{\rm thermal}}{P_{\rm magnetic}}.
#
# With different values of :math:`\beta`, the temperature structure and various other thermodynamic properties
# of the clusters will vary.
#
# To get started, we'll need to import the relevant building blocks.

import tempfile

import numpy as np
import unyt

from pisces.models.galaxy_clusters import MagnetizedSphericalGalaxyClusterModel

# Import the model class and the construction profiles.
from pisces.profiles import NFWDensityProfile

# %%
# Profiles
# ''''''''
# To start building the models, we'll need to create the initial profiles. A quick look at the documentation
# for the model shows that the full cluster model can be built from a **total density profile** :math:`\rho_{\rm tot}` and
# a **gas density profile** :math:`\rho_{\rm gas}`. For realism, we'll use a gas fraction close to ``0.1``, which is
# pretty standard for most clusters. We'll also use an NFW core density around :math:`5 \times 10^6 \;{\rm M_\odot\; kpc^{-3}}`.

# Construct the total profile.
total_density_profile = NFWDensityProfile(
    rho_0=unyt.unyt_quantity(5e6, "Msun/kpc**3"), r_s=unyt.unyt_quantity(200, "kpc")
)

# Construct the gas density profile.
gas_density_profile = NFWDensityProfile(
    rho_0=unyt.unyt_quantity(5e5, "Msun/kpc**3"), r_s=unyt.unyt_quantity(220, "kpc")
)

# %%
# Model Construction
# ------------------
# Now that we have the density profiles, it's an easy matter to generate the relevant profiles using
# the :meth:`~models.galaxy_clusters.spherical.MagnetizedSphericalGalaxyClusterModel.from_density_and_total_density`
# method. We'll store the data files in a temporary directory and create a number of models for different values
# of the :math:`\beta` parameter.

# Create a temp directory.
tmpdir = tempfile.TemporaryDirectory()

# Define the selection of beta values
beta_set = [np.inf, 1000, 500, 200, 100, 50, 10, 5, 1]

# Set some parameters.
rmin, rmax = unyt.unyt_quantity(1, "kpc"), unyt.unyt_quantity(5, "Mpc")

models = {}

for beta_id, beta in enumerate(beta_set):
    # Determine the filename.
    filename = f"{tmpdir.name}/model_beta_{beta_id}.h5"

    # Create the model
    models[beta_id] = MagnetizedSphericalGalaxyClusterModel.from_density_and_total_density(
        gas_density_profile,
        total_density_profile,
        filename,
        min_radius=rmin,
        max_radius=rmax,
        num_points=500,
        overwrite=True,
        beta_profile=beta,  # This sets the beta value.
    )

# %%
# Extract and Plot Results
# ------------------------
# Once the models are generated, we can extract their radial profiles and visualize how magnetic
# support modifies the cluster structure. We'll focus on temperature, entropy, magnetic field strength,
# and Alfvén velocity.
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams["xtick.major.size"] = 8
plt.rcParams["xtick.minor.size"] = 5
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# Fields to extract and plot
fields_to_plot = ["temperature", "entropy", "magnetic_field", "alfven_velocity"]
field_units = ["keV", "keV*cm**2", "G", "km/s"]
field_labels = {
    "temperature": r"Temperature [$\mathrm{keV}$]",
    "entropy": r"Entropy [$\mathrm{keV\ cm^2}$]",
    "magnetic_field": r"Magnetic Field [$\mathrm{G}$]",
    "alfven_velocity": r"Alfvén Velocity [$\mathrm{km/s}$]",
}

# Setup the figures.
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
axes = axes.flatten()

norm = LogNorm(vmin=np.amin(beta_set), vmax=np.amax(beta_set[1:]))
colors = plt.cm.plasma(norm(np.asarray(beta_set)))

# Cycle through each plot and then each model and plot
# the results.
for ax, field, units in zip(axes, fields_to_plot, field_units):
    label = field_labels[field]

    for beta_id, model in models.items():
        beta = beta_set[beta_id]

        ax.plot(model.grid["r"].d, model[field].to_value(units), lw=2, color=colors[beta_id])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(label)

axes[-2].set_xlabel(r"Radius [kpc]")
axes[-1].set_xlabel(r"Radius [kpc]")

# Create a scalar mappable for the colorbar
import matplotlib as mpl

# Normalize and colormap (log-scale)
sm = mpl.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
sm.set_array([])

# Create the colorbar with a triangle tip on the upper end
cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.02, extend="max")
cbar.set_label(r"$\beta$")

plt.show()
