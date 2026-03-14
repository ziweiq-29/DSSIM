from skimage.metrics import structural_similarity as ssim
import numpy as np

orig = np.fromfile("original.dat", dtype=np.float32).reshape(360,180,26)
dec  = np.fromfile("decompressed.dat", dtype=np.float32).reshape(360,180,26)

# normalization
smin = min(orig.min(), dec.min())
smax = max(orig.max(), dec.max())

orig = (orig - smin) / (smax - smin)
dec  = (dec  - smin) / (smax - smin)

# SSIM
ssim_val = ssim(orig, dec, data_range=1.0)

# dSSIM
dssim = (1 - ssim_val) / 2

print("dssim =", dssim)