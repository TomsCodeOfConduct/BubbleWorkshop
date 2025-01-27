from math import sqrt
from math import pi
import numpy as np # type: ignore
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import windows

# Step 1: Load the WAV file
fs, data = wavfile.read("C:/Users/reabt/OneDrive/Documents/Hackathon Stuff/Data/2019-05-21T15-00-12.wav")  # Replace with the correct path to your WAV file

# Step 2: Define parameters
tstart = 10  # Starting time in seconds
ts = 5  # Time segment in seconds
bubbleRadius = 0
poly = 1.4
rho = 1025
g = 9.81
d = 120 
patm = 101325
pst = patm + (rho*d*g)

# If the data has multiple channels (e.g., stereo or multi-channel), choose one (e.g., first channel)
# If data is 1D, it's already a single-channel signal
data = data[:, 0] if len(data.shape) > 1 else data

# Define the time vector for ts seconds of data
t = np.arange(tstart * fs, (tstart + ts) * fs) / fs  # Time vector for the segment of interest

# Get the length of the data segment
short = data[tstart * fs: (tstart + ts) * fs]  # Segment of data for tstart to tstart+ts seconds

# H is the number of channels or hydrophones (columns in the data)
#H = data.shape[1] if len(data.shape) > 1 else 1
H = 5
L = len(short)  # Number of samples

# Segment parameters
len_seg = 512  # Length of each segment
overlap = 256  # Overlap between segments
N_s = len(t) // overlap - 1  # Number of segments

# Step 3: Create a function to extract segments
def seg(p, k):
    return short[(p - 1) * overlap: (p - 1) * overlap + len_seg]

# Step 4: Create the FFT function with Hamming window
def X_pk(k, p):
    return np.fft.fft(windows.hamming(len(seg(p, k))) * seg(p, k), 9600)  # 9600 is the number of FFT points

# Frequency vector
Fs = fs / len(X_pk(1, 1)) * np.arange(len(X_pk(1, 1)))

# Step 5: Cross-spectral function
def C_kl(k, l, p):
    return np.abs(np.conj(X_pk(k, p)) * X_pk(l, p))

# Step 6: Initialize arrays for cross-spectrogram calculation
barC_hold = np.zeros_like(C_kl(1, 1, 1))
barC = np.zeros((N_s, len(C_kl(1, 1, 1))))

# Step 7: Loop through segments to calculate cross-spectrogram
for pc in range(1, N_s + 1):
    barC_hold = np.zeros_like(C_kl(1, 1, 1))
    
    # Calculate cross-spectrogram only if H > 1 (i.e., multiple hydrophones)
    if H > 1:
        for kc in range(1, H):
            for lc in range(kc + 1, H + 1):
                barC_hold += C_kl(kc, lc, pc)
        
        barC[pc - 1, :] = 2 / (H * (H - 1)) * barC_hold
    else:
        # Handle the case when H = 1 (only one channel), no cross-spectrogram can be computed
        barC[pc - 1, :] = np.zeros_like(barC_hold)

# Step 8: Convert to dB with a larger minimum value to avoid very small numbers
# Ensure there are no zeros before log10 conversion
barC_safe = np.maximum(barC, 1e-6)  # Ensure there are no zeros in barC
barCdB = 10 * np.log10(barC_safe)  # Convert to dB

# Step 9: Sum over segments
sumC = np.mean(barCdB, axis=0)

# Step 10: Normalize each segment
hatC = barCdB - sumC

# Check for NaN or infinite values in hatC, and replace them with zeros if they exist
hatC = np.nan_to_num(hatC, nan=0.0, posinf=0.0, neginf=0.0)

# Time vector for plotting
t = np.linspace(0, ts, N_s)

# Step 11: Plotting results
fig, axs = plt.subplots(4, 1, figsize=(10, 8))

# Cross Spectrogram
axs[0].imshow(barCdB.T, aspect='auto', origin='lower', extent=[0, ts, 0, 12000], cmap='viridis')  # Set color map to 'viridis'
axs[0].set_title(f'Cross Spectrogram of {ts}s of data')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Frequency [Hz]')
fig.colorbar(axs[0].imshow(barCdB.T, aspect='auto', origin='lower', extent=[0, ts, 0, 12000], cmap='viridis'), ax=axs[0])

# Normalized Cross Spectrogram
axs[1].imshow(hatC.T, aspect='auto', origin='lower', extent=[0, ts, 0, 12000], cmap='plasma')  # Set color map to 'plasma'
axs[1].set_title(f'Normalized Cross Spectrogram of {ts}s of data')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Frequency [Hz]')
fig.colorbar(axs[1].imshow(hatC.T, aspect='auto', origin='lower', extent=[0, ts, 0, 12000], cmap='plasma'), ax=axs[1])

# Thresholded Cross Spectrogram
Th = 10
hatC_ThdB = hatC.copy()
hatC_ThdB[hatC <= Th] = 0

axs[2].imshow(hatC_ThdB.T, aspect='auto', origin='lower', extent=[0, ts, 0, 12000], cmap='inferno')  # Set color map to 'inferno'
axs[2].set_title(f'Thresholded at {Th} dB')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Frequency [Hz]')
fig.colorbar(axs[2].imshow(hatC_ThdB.T, aspect='auto', origin='lower', extent=[0, ts, 0, 12000], cmap='inferno'), ax=axs[2])

row = 0
bubbleCount = 0
bubbleFlag = False
bubbleList = []

for line in hatC_ThdB:
    row += 1
    col = 0
    bubbleSize = 0
    for val in line:
        col += 1
        if(val > 0):
            bubbleSize += 1
            if(bubbleSize == 10):
                bubbleCount += 1
                bubbleFlag = True
        elif bubbleFlag == True and val == 0:
            f_max = col - 1
            bubbleFlag = False
            if(f_max != 0):
                bubbleRadius = (1 / (2*pi*f_max)) * ( sqrt( (3*poly*pst)/rho ) ) 
                bubbleList.append(bubbleRadius)

print(row)
print(col)
print("Total bubbles: " , bubbleCount)

# Define the bins and range for the histogram
bins = 200  # Number of bins
range_min = 0  # Minimum value for x-axis
range_max = 0.1  # Maximum value for x-axis

# Create the histogram
axs[3].hist(bubbleList, bins=bins, range=(range_min, range_max), edgecolor='black')

# Set title and labels
axs[3].set_title("Bubble Size Distribution")
axs[3].set_xlabel("Bubble Size (cm)")
axs[3].set_ylabel("Bubble Count")

# Adjust x-axis scale to be from 0 to 0.1 (this is the same as `range=(0, 0.1)`)
axs[3].set_xlim([0, 0.1])  # Limit the x-axis to range from 0 to 0.1
axs[3].set_ylim([0, 150])  # Adjust the y-axis range as needed


plt.tight_layout()
plt.show()