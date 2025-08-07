import matplotlib.pyplot as plt
from pisces.profiles.density import (
    PlummerDensityProfile,
)
#
r = np.linspace(0.1, 10, 100)
profile = PlummerDensityProfile(M=1.0, r_s=1.0)
rho = profile(r)
#
_ = plt.loglog(
    r, rho, "k-", label="Plummer Profile"
)
_ = plt.xlabel("Radius (r)")
_ = plt.ylabel("Density (rho)")
_ = plt.legend()
plt.show()
