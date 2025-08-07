import matplotlib.pyplot as plt
from pisces.profiles.density import (
    VikhlininDensityProfile,
)
#
r = np.linspace(0.1, 10, 100)
profile = VikhlininDensityProfile(
    rho_0=1.0,
    r_c=1.0,
    r_s=5.0,
    alpha=1.0,
    beta=1.0,
    epsilon=1.0,
    gamma=3.0,
)
rho = profile(r)
#
_ = plt.loglog(
    r, rho, "k-", label="Vikhlinin Profile"
)
_ = plt.xlabel("Radius (r)")
_ = plt.ylabel("Density (rho)")
_ = plt.legend()
plt.show()
