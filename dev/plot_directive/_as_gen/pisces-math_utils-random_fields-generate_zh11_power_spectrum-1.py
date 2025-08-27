from pisces.math_utils.random_fields import generate_zh11_power_spectrum
L, n = 10.0, 256
k_grid = np.fft.fftfreq(n, d=L/n) * 2 * np.pi
k_mag = np.abs(k_grid)
ps_func = generate_zh11_power_spectrum(amplitude=1.0, k0=5.0, k1=0.5)
ps_array = ps_func(k_mag)

import matplotlib.pyplot as plt
plt.loglog(k_mag[1:], ps_array[1:])  # skip k=0
plt.xlabel('Wavenumber k')
plt.ylabel('Power Spectrum P(k)')
plt.title('ZuHone+11 Spectrum with Cutoffs')
plt.show()