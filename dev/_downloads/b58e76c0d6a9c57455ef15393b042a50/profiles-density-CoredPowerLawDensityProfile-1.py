import matplotlib.pyplot as plt
from pisces.profiles.density import (
    CoredPowerLawDensityProfile,
)
#
r = np.logspace(-1, 1, 100)
profile = CoredPowerLawDensityProfile(
    rho_0=1.0, r_c=1.0, gamma=2.0
)
rho = profile(r)
#
plt.loglog(r, rho, label="Cored Power-Law")
plt.xlabel("Radius (r)")
plt.ylabel("Density (rho)")
plt.legend()
plt.show()
