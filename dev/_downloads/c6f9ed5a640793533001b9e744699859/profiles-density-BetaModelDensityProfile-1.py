import matplotlib.pyplot as plt
from pisces.profiles.density import (
    BetaModelDensityProfile,
)
#
r = np.logspace(-1, 1, 100)
profile = BetaModelDensityProfile(
    rho_0=1.0, r_c=1.0, beta=2 / 3
)
rho = profile(r)
#
_ = plt.loglog(r, rho, label="Beta Model")
_ = plt.xlabel("Radius (r)")
_ = plt.ylabel("Density (rho)")
_ = plt.legend()
plt.show()
