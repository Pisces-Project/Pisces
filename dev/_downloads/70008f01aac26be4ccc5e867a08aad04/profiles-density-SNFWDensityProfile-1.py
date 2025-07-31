import matplotlib.pyplot as plt
from pisces.profiles.density import (
    SNFWDensityProfile,
)
#
r = np.linspace(0.1, 10, 100)
profile = SNFWDensityProfile(M=1.0, a=1.0)
rho = profile(r)
#
_ = plt.loglog(r, rho, "k-", label="SNFW Profile")
_ = plt.xlabel("Radius (r)")
_ = plt.ylabel("Density (rho)")
_ = plt.legend()
plt.show()
