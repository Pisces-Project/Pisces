from pisces.profiles.density import (
    FlaredDiskDensityProfile,
)
r, z = np.meshgrid(
    np.linspace(0, 15, 200),
    np.linspace(-3, 3, 200),
)
profile = FlaredDiskDensityProfile(
    rho_0=1.0,
    r_s=3.0,
    z_0=0.3,
    r_f=5.0,
    delta=0.5,
)
rho = profile(r, z)
