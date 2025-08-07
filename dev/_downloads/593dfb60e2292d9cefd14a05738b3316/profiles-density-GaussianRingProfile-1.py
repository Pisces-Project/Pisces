from pisces.profiles.density import (
    GaussianRingProfile,
)
import matplotlib.pyplot as plt
R = np.linspace(0, 10, 200)
profile = GaussianRingProfile(
    Sigma_0=1.0, R_0=5.0, sigma=0.5
)
Sigma = profile(R)
plt.plot(R, Sigma)
plt.xlabel("Radius (R)")
plt.ylabel("Surface Density (Sigma)")
plt.title("Gaussian Ring Profile")
plt.grid(True)
plt.show()
