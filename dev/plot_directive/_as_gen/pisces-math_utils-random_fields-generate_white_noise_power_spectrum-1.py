from pisces.math_utils.random_fields import generate_white_noise_power_spectrum
L, n = 10.0, 128
k_grid = np.fft.fftfreq(n, d=L/n) * 2 * np.pi
power_spectrum = generate_white_noise_power_spectrum()
ps_array = power_spectrum(k_grid)

import matplotlib.pyplot as plt
plt.loglog(k_grid, ps_array)
plt.xlabel('Wavenumber k')
plt.ylabel('Power Spectrum P(k)')
plt.title('White Noise Power Spectrum')
plt.show()