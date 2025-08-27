import matplotlib.pyplot as plt
from pisces.profiles.density import (
    EinastoDensityProfile,
)
#
r = np.linspace(0.1, 10, 100)
profile = EinastoDensityProfile(
    rho_0=1.0, r_s=1.0, alpha=0.18
)
rho = profile(r)
#
_ = plt.loglog(
    r, rho, "k-", label="Einasto Profile"
)
_ = plt.xlabel("Radius (r)")
_ = plt.ylabel("Density (rho)")
_ = plt.legend()
plt.show()
