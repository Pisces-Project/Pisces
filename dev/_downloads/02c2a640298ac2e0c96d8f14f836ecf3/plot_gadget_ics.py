"""
====================================
Initial Conditions for Gadget-4
====================================

This example demonstrates how to generate ICs for
Gadget-4. As an example, we'll create two spherical galaxy
cluster models with slightly different properties and then construct
initial conditions where they will collide. These initial conditions
can then be passed off to Gadget-4 to run the simulation.

Overview
---------
This example connects many parts of the Pisces framework in order to go from
the raw profiles used to build the models all the way to the simulation ready initial
conditions. As such, we'll break the example down into a few key steps:

1. **Model Construction**: This will involve taking profiles from :mod:`~profiles` and
   using the :class:`~pisces.models.galaxy_clusters.spherical.SphericalGalaxyClusterModel` to build
   hydrodynamic models of galaxy clusters.
2. **Initial Conditions**: One the models are generated, we'll create the initial conditions object
   and align / position the clusters in 3D space so that they are ready to be simulated.
3. **Particle Generation**: In order to create Gadget-4 compatible ICs, we'll need to sample particles
   from the hydrodynamic models. This will be done using the
   :meth:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions.generate_particles` method.
4. **Export**: Finally, we'll export the initial conditions to a Gadget-4 compatible HDF5 file using the
   :class:`~pisces.extensions.simulation.gadget.frontends.Gadget4Frontend` class.

.. contents::

Step 1: Model Construction
---------------------------
For this example, we'll create simple toy models for the galaxy clusters using NFW profiles for both
the total and gas densities (:class:`~profiles.density.NFWDensityProfile`). The clusters will be non-magnetized
and assume pure thermal pressure support.
"""

# %%
# Setup
# ^^^^^^
# We begin by importing the required profiles and model class. This example uses:
#
# - :class:`~pisces.models.galaxy_clusters.spherical.SphericalGalaxyClusterModel`
# - :class:`~profiles.density.NFWDensityProfile`
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import unyt

from pisces.models.galaxy_clusters import SphericalGalaxyClusterModel
from pisces.profiles import NFWDensityProfile

# %%
# Density Profiles
# ^^^^^^^^^^^^^^^^
# We'll define NFW profiles for both the total matter and gas components. The selected
# central densities and scale radii are typical for galaxy clusters, but can be adjusted
# to explore different cluster properties. These values should not be taken as endorsed
# values by the authors.

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
# ^^^^^^^^^^^^^^^^^^^^
# The next step is to bring everything together and make the spherical galaxy cluster model
# that we're going to use in the simulation. We'll use a logarithmic grid to get good resolution
# near the core of the cluster while still extending out to large radii.

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
# Step 2: Initial Conditions
# ---------------------------
# Now that the model is built, we want to place two copies of the model into 3D space so
# that they are ready to be simulated in a collision. To do this, we'll use the
# :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions` class,
# which effectively acts as a container for the models and the particles that will be
# sampled from them. If you're unfamiliar with this class, you can read more about it
# in the :ref:`initial_conditions_overview` document.
#
# In this case, we'll probably want an 10 Mpc box to contain both of the clusters, so we'll
# put the first one at :math:`(3, 5, 5)` Mpc and the second at :math:`(7, 5, 5)` Mpc. Then we'll
# give them each a kick of around :math:`1000 \;{\rm km \; s^{-1}}` towards each other along the x-axis.

# Import the initial conditions class.
from pisces.extensions.simulation.core import InitialConditions3DCartesian

# Create the model configuration tuples to tell
# the ICs where the models are and how they are moving.
models = [
    {
        "model_name": "cluster_1",
        "model": model,
        "position": unyt.unyt_array([3.0, 5.0, 5.0], "Mpc"),
        "velocity": unyt.unyt_array([1000.0, 0.0, 0.0], "km/s"),
    },
    {
        "model_name": "cluster_2",
        "model": model,
        "position": unyt.unyt_array([7.0, 5.0, 5.0], "Mpc"),
        "velocity": unyt.unyt_array([-1000.0, 0.0, 0.0], "km/s"),
    },
]

# Create the initial conditions object.
directory = f"{tmpdir.name}/gadget_ics"
ics = InitialConditions3DCartesian.create_ics(
    directory,
    *models,
)

# %%
# Step 3: Particle Generation
# ---------------------------
# Now we need to create the particles that we want to include in the simulation. For
# the sake of runtime for this example, we'll just create 10,000 particles per cluster,
# but in a real simulation you'd probably want to use many more than that. The
# :meth:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions.generate_particles` method
# attaches to the model's own sampling routines to create the particles, so it should
# be fairly efficient and straightforward to use.

# define the number of particles
num_particles = {
    "gas": 10_000,
    "dark_matter": 10_000,
}

# Generate the particles
ics.generate_particles("cluster_1", num_particles, overwrite=True)
ics.generate_particles("cluster_2", num_particles, overwrite=True)

# %%
# Step 4: Export to Gadget-4
# ---------------------------
# The final step is to export the initial conditions to a Gadget-4 compatible HDF5 file.
# This is done using the :class:`~pisces.extensions.simulation.gadget.frontends.Gadget4Frontend` class,
# which handles the details of writing the file in a way that Gadget-4 can read it.
#
# We'll use a box with lengths of 10 Mpc on each side and we'll set up the unit system
# so that we have lengths in kpc, masses in :math:`{\rm M_\odot}`, velocities in :math:`{\rm km \; s^{-1}}`.

# Import the frontend
from pisces.extensions.simulation.gadget import Gadget4Frontend

# Create the frontend attachment
frontend = Gadget4Frontend(ics)

# Modify the box size and the units so that everything is
# made self-consistent.
frontend.config["parameters.boxsize"] = unyt.unyt_quantity(10, "Mpc")
frontend.config["units.length"] = unyt.Unit("kpc")
frontend.config["units.mass"] = unyt.Unit("Msun")
frontend.config["units.velocity"] = unyt.Unit("km/s")

# Now create the initial conditions file at
# gadget_ics.hdf5
ic_particles = frontend.generate_initial_conditions(f"{tmpdir.name}/gadget_ics.hdf5", overwrite=True)

# %%
# Visualization
# --------------
# Finally, let's visualize the projected positions of the gas particles
# from both clusters in the x–y plane. We'll also color by particle density
# using a 2D histogram.

fig, ax = plt.subplots(figsize=(8, 5))

# Extract gas particle positions (PartType0 = gas)
coords = ic_particles["ParticleType0.Coordinates"]
x, y = coords[:, 0], coords[:, 1]

# 2D hexbin plot of particle distribution
hb = ax.hexbin(
    x,
    y,
    gridsize=150,
    cmap="plasma",
    bins="log",
    mincnt=1,
)
ax.set_aspect("equal")

# Labels and colorbar
ax.set_xlabel(r"$x \; [{\rm kpc}]$")
ax.set_ylabel(r"$y \; [{\rm kpc}]$")
cb = fig.colorbar(hb, ax=ax, label="log(N particles)", fraction=0.05)

ax.set_title("Projected Gas Particle Distribution (x–y plane)")
plt.tight_layout()
plt.show()
