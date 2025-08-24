from pisces.profiles.temperature import (
    IsothermalTemperatureProfile,
)
import matplotlib.pyplot as plt
r = np.linspace(0, 10, 100)
profile = IsothermalTemperatureProfile(
    T_0=1e7 * unyt.Unit("K")
)
T = profile(r)
plt.plot(r, T)
plt.xlabel("r [kpc]")
plt.ylabel("Temperature [K]")
plt.grid()
plt.show()
