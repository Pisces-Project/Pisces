import matplotlib.pyplot as plt
from pisces.profiles.density import (
    GeneralizedNFWDensityProfile,
)
#
r = np.logspace(-1, 1, 100)
profile = GeneralizedNFWDensityProfile(
    rho_0=1.0,
    r_s=1.0,
    alpha=1.0,
    beta=3.0,
    gamma=1.0,
)
rho = profile(r)
#
plt.loglog(r, rho, label="GNFW Profile")
plt.xlabel("Radius (r)")
plt.ylabel("Density (rho)")
plt.legend()
plt.show()
