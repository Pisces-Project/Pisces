from pisces.profiles.density import (
    CoreSersicProfile,
)
import matplotlib.pyplot as plt
R = np.linspace(0.1, 20, 300)
profile = CoreSersicProfile(
    Sigma_b=1.0,
    R_b=2.0,
    R_e=5.0,
    n=4.0,
    gamma=0.5,
    alpha=5.0,
)
Sigma = profile(R)
plt.semilogy(R, Sigma)
plt.xlabel("Radius (R)")
plt.ylabel("Surface Density (Sigma)")
plt.title("Core-Sérsic Surface Profile")
plt.grid(True)
plt.show()
