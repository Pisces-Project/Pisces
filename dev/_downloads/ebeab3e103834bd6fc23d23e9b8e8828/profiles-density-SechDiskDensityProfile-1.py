from pisces.profiles.density import (
    SechDiskDensityProfile,
)
r, z = np.meshgrid(
    np.linspace(0, 10, 100),
    np.linspace(-2, 2, 100),
)
profile = SechDiskDensityProfile(
    rho_0=1.0, r_s=3.0, z_s=0.3
)
rho = profile(r, z)
