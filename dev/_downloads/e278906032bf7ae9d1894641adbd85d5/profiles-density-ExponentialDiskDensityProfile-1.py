import matplotlib.pyplot as plt
from pisces.profiles.density import (
    ExponentialDiskDensityProfile,
)
#
r = np.linspace(0, 15, 200)
z = 0.0  # Midplane
#
profile = ExponentialDiskDensityProfile(
    rho_0=1.0, r_s=3.0, z_s=0.3
)
rho = profile(r, z)
#
plt.semilogy(
    r, rho, label="Exponential Disk (z=0)"
)
plt.xlabel("Radius (r)")
plt.ylabel("Density (rho)")
plt.legend()
plt.show()
