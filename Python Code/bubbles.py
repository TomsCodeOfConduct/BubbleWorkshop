import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
from scipy.fft import fft

# Given values
tstart = 10  # seconds of interest
ts = 5  # seconds of interest
fs = 96000  # sampling frequency
H = len(data[0])  # number of hydrophones
short = data[tstart * fs : (tstart + ts) * fs, :]  # shorten data over time period of interest
L = len(short[:, 0])  # number of samples

len_seg = 512  # segment length
overlap = 256  # overlap between segments
N_s = len(t) // 256 - 1  # number of segments

# Create segment function
def seg(p, k):
    return short[(p - 1) * overlap : (p - 1) * overlap + len_seg, k]

# FFT function with Hamming window
def X_pk(k, p):
    return fft(hamming(len(seg(p, k))) * seg(p, k), 9600)

# Frequency vector
Fs = np.linspace(0, fs / len(X_pk(1, 1)) * (len(X_pk(1, 1)) - 1), len(X_pk(1, 1)))

# Cross-spectrum function
def C_kl(k, l, p):
    return np.abs(np.conj(X_pk(k, p)) * X_pk(l, p))

# Initialize variables
barC_hold = np.zeros_like(C_kl(1, 1, 1))
barC = np.zeros((N_s, len(C_kl(1, 1, 1))))

# Loop to calculate cross-spectrogram
for pc in range(N_s):
    barC_hold = np.zeros_like(C_kl(1, 1, 1))
    for kc in range(H - 1):
        for lc in range(kc + 1, H):
            barC_hold += C_kl(kc, lc, pc)
    barC[pc, :] = 2 / (H * (H - 1)) * barC_hold

# Convert to dB
barCdB = 10 * np.log10(barC / 1e-6)

# Sum over segments
sumC = np.mean(barCdB, axis=0)

# Normalize each segment data
hatC = barCdB - sumC

# Time vector for plotting
t = np.linspace(0, ts, N_s)

# Plotting
plt.figure(figsize=(10, 12))

# Plot Cross Spectrogram
plt.subplot(3, 1, 1)
plt.surf(t, Fs, barCdB.T, edgecolor='none')
plt.view_init(azim=0, elev=90)
plt.colorbar()
plt.ylim([0, 12000])
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.title(f'Cross Spectrogram of {ts}s of data')

# Plot Normalized Cross Spectrogram
plt.subplot(3, 1, 2)
plt.surf(t, Fs, hatC.T, edgecolor='none')
plt.view_init(azim=0, elev=90)
plt.colorbar()
plt.ylim([0, 12000])
plt.clim([0, 20])
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.title(f'Normalized Cross Spectrogram of {ts}s of data')

# Apply threshold
Th = 10
Th_db = np.where(hatC <= Th)
bubbles = np.where(hatC >= Th)

hatC_ThdB = hatC.copy()
hatC_ThdB[Th_db] = 0

# Plot Thresholded Spectrogram
plt.subplot(3, 1, 3)
plt.surf(t, Fs, hatC_ThdB.T, edgecolor='none')
plt.view_init(azim=0, elev=90)
plt.colorbar()
plt.ylim([0, 12000])
plt.clim([10, 20])
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.title(f'Thresholded at {Th} dB')
plt.colormap('jet')

plt.tight_layout()
plt.show()

# Bubble radius calculation (optional part based on further data)
f_max = ...  # maximum energy frequency bin (to be defined)
poly = ...  # polytropic index of gas (to be defined)
Pst = ...  # Pressure (Pa) (to be defined)
rho = 1025  # seawater density (kg/m^3)

# Assuming Patm (atmospheric pressure) and Pst (pressure) are given
Pst = Patm + ...

# Radius calculation
R0 = (1 / (2 * np.pi * f_max)) * np.sqrt((3 * poly * Pst) / rho)
