import matplotlib.pyplot as plt
from pisces.profiles.density import (
    KingDensityProfile,
)
#
r = np.linspace(0.1, 10, 100)
profile = KingDensityProfile(
    rho_0=1.0, r_c=1.0, r_t=5.0
)
rho = profile(r)
#
_ = plt.loglog(r, rho, "k-", label="King Profile")
_ = plt.xlabel("Radius (r)")
_ = plt.ylabel("Density (rho)")
_ = plt.legend()
plt.show()
