import matplotlib.pyplot as plt
from pisces.profiles.density import (
    TNFWDensityProfile,
)
#
r = np.linspace(0.1, 10, 100)
profile = TNFWDensityProfile(
    rho_0=1.0, r_s=1.0, r_t=10.0
)
rho = profile(r)
#
_ = plt.loglog(r, rho, "k-", label="TNFW Profile")
_ = plt.xlabel("Radius (r)")
_ = plt.ylabel("Density (rho)")
_ = plt.legend()
plt.show()
