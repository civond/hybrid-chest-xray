# Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# open the image
f = cv2.imread('Data\covid\img\COVID-4.png',0)
initial = f
clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
f = clahe.apply(f)
f_orig = f
# transform image into freq. domain and shifted
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

plt.imshow(np.log1p(np.abs(Fshift)), cmap='gray')
plt.axis('off')
#plt.show()

# Butterwort Low Pass Filter
M,N = f.shape
D0_low = 30 # cut of frequency for low-pass filter
n = 3 # order 
H1 = np.zeros((M,N), dtype=np.float32)
for u in range(M):
    for v in range(N):
        D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        H1[u,v] = 1 / (1 + (D/D0_low)**n)
        
plt.imshow(H1, cmap='gray')
plt.axis('off')
plt.title("Low-pass filter")
#plt.show()

# Butterworth High Pass Filter
D0_high = 15 # cut of frequency for high-pass filter
n = 3
H2 = np.zeros((M,N), dtype=np.float32)
for u in range(M):
    for v in range(N):
        D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        H2[u,v] = 1 / (1 + (D0_high/D)**n)
        
plt.imshow(H2, cmap='gray')
plt.axis('off')
plt.title("High-pass filter")
#plt.show()

# Combine low-pass and high-pass filters to create band-pass filter
H_bandpass = H1 * H2
plt.figure(1, figsize=(4,4))
plt.imshow(H_bandpass, cmap='gray')
plt.axis('off')
plt.title("Bandpass Freq. Resp.")
plt.show()

# Apply band-pass filter to the frequency domain
Gshift = Fshift * H_bandpass
G = np.fft.ifftshift(Gshift)
g_bandpass = np.abs(np.fft.ifft2(G))
test = f-g_bandpass
cv2.imwrite("result.png", test)
plt.figure(2)
plt.subplot(3,1,1)
plt.imshow(initial, cmap='gray')
plt.title("Initial")
plt.axis('off')

plt.subplot(3,1,2)
plt.imshow(f_orig, cmap='gray')
plt.title("CLAFE")
plt.axis('off')

plt.subplot(3,1,3)
plt.imshow(g_bandpass, cmap='gray')
plt.title("Bandpass")
plt.axis('off')
plt.show()
