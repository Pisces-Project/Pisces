from pisces.math_utils.random_fields import generate_gaussian_cutoff_power_spectrum
L, n = 10.0, 256
k_grid = np.fft.fftfreq(n, d=L/n) * 2 * np.pi
k_mag = np.abs(k_grid)
power_spectrum = generate_gaussian_cutoff_power_spectrum(k0=5.0, sigma=2.0)
ps_array = power_spectrum(k_mag)

import matplotlib.pyplot as plt
plt.plot(k_mag, ps_array)
plt.xlabel('Wavenumber k')
plt.ylabel('Power Spectrum P(k)')
plt.title('Gaussian-Cutoff Spectrum')
plt.show()