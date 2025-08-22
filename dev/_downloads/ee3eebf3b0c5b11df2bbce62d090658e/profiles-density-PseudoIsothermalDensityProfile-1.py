import matplotlib.pyplot as plt
from pisces.profiles.density import (
    PseudoIsothermalDensityProfile,
)
#
r = np.logspace(-1, 1, 100)
profile = PseudoIsothermalDensityProfile(
    rho_0=1.0, r_c=1.0
)
rho = profile(r)
#
plt.loglog(r, rho, label="PISO Profile")
plt.xlabel("Radius (r)")
plt.ylabel("Density (rho)")
plt.legend()
plt.show()
