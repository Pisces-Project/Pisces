import matplotlib.pyplot as plt
from pisces.profiles.density import (
    AM06DensityProfile,
)
#
r = np.linspace(0.1, 10, 100)
profile = AM06DensityProfile(
    rho_0=1.0,
    a_c=1.0,
    c=2.0,
    a=3.0,
    alpha=1.0,
    beta=3.0,
)
rho = profile(r)
#
_ = plt.loglog(r, rho, "k-", label="AM06 Profile")
_ = plt.xlabel("Radius (r)")
_ = plt.ylabel("Density (rho)")
_ = plt.legend()
plt.show()
