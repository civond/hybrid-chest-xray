import cv2
import numpy as np
import scipy.signal as signal
import os
import matplotlib.pyplot as plt

datadir = "Data/"
classes = os.listdir(datadir)

# Preprocessing
for folder in classes:
    img_dir = os.path.join(datadir, folder,'img/')
    dir_contents = os.listdir(img_dir)
    for content in dir_contents:
        img_path = os.path.join(img_dir,content)
        image = cv2.imread(img_path, 0)
        temp = image
        print(f"Converting: {img_path}")

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
        image = clahe.apply(image)

        # Butterworth
        F = np.fft.fft2(image)
        Fshift = np.fft.fftshift(F)

        [M,N] = image.shape
        order = 3

        #Cutoff Frequencies
        D0_low = 30
        D0_high = 15

        # Low Pass
        H1 = np.zeros((M,N), dtype=np.float32)
        for u in range(M):
            for v in range(N):
                D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
                H1[u,v] = 1 / (1 + (D/D0_low)**order)
        #High Pass
        H2 = np.zeros((M,N), dtype=np.float32)
        for u in range(M):
            for v in range(N):
                D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
                H2[u,v] = 1 / (1 + (D0_high/D)**order)
                
        # Combining filters
        H_bandpass = H1 * H2
        Gshift = Fshift * H_bandpass
        G = np.fft.ifftshift(Gshift)
        g_bandpass = np.abs(np.fft.ifft2(G) )
        result = image-g_bandpass
        result = np.clip(result, 0, 255).astype(np.uint8)

        cv2.imwrite(img_path, result)

        """cv2.imshow("Orig", temp)
        cv2.imshow("CLAHE", image)
        cv2.imshow("BP", g_bandpass)
        cv2.imshow("res", result)
        cv2.waitKey(0)"""