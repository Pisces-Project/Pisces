from pisces.profiles.density import SersicProfile
import matplotlib.pyplot as plt
R = np.linspace(0.01, 10, 200)
profile = SersicProfile(
    Sigma_0=1.0, R_e=2.0, n=4.0
)
Sigma = profile(R)
plt.semilogy(R, Sigma)
plt.xlabel("Radius (R)")
plt.ylabel("Surface Density (Sigma)")
plt.title("Sérsic Surface Profile (n=4)")
plt.grid(True)
plt.show()
