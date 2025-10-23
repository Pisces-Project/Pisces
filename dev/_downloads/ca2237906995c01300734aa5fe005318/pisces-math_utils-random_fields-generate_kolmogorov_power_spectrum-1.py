from pisces.math_utils.random_fields import generate_kolmogorov_power_spectrum
L, n = 10.0, 256
k_grid = np.fft.fftfreq(n, d=L/n) * 2 * np.pi
k_mag = np.abs(k_grid)
power_spectrum = generate_kolmogorov_power_spectrum(amplitude=1.0, k0=1.0, k_cut=20.0)
ps_array = power_spectrum(k_mag)

import matplotlib.pyplot as plt
plt.loglog(k_mag[1:], ps_array[1:])  # skip k=0
plt.xlabel('Wavenumber k')
plt.ylabel('Power Spectrum P(k)')
plt.title('Kolmogorov-like Spectrum with Cutoff')
plt.show()