import matplotlib.pyplot as plt
from pisces.profiles.density import (
    SingularIsothermalDensityProfile,
)
#
r = np.linspace(0.1, 10, 100)
profile = SingularIsothermalDensityProfile(
    rho_0=1.0
)
rho = profile(r)
#
_ = plt.loglog(r, rho, "k-", label="SIS Profile")
_ = plt.xlabel("Radius (r)")
_ = plt.ylabel("Density (rho)")
_ = plt.legend()
plt.show()
