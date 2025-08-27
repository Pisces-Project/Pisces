from pisces.math_utils.random_fields import generate_power_law_power_spectrum
L, n = 10.0, 128
k_grid = np.fft.fftfreq(n, d=L/n) * 2 * np.pi
k_mag = np.abs(k_grid)
power_spectrum = generate_power_law_power_spectrum(slope=-11/3)
ps_array = power_spectrum(k_mag)

import matplotlib.pyplot as plt
plt.loglog(k_mag[1:], ps_array[1:])  # skip k=0
plt.xlabel('Wavenumber k')
plt.ylabel('Power Spectrum P(k)')
plt.title('Power-Law Spectrum (slope = -11/3)')
plt.show()