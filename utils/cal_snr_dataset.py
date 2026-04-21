from PIL import Image
import numpy as np
import os
def calculate_snr(image):

    image = np.array(image)
    signal = np.mean(image/255)
    # 计算图像噪声
    noise = np.std(image/255)
    noise = np.mean((image/255 - np.mean(image/255)) ** 2)
    # 计算 SNR
    snr = 20 * np.log10(signal / noise)
    return snr

# 读取图像
dataset_path= r'/home/yuanmaoxun/Datasets/LLVIP/test'
snr_values_rgb = []
snr_values_lwir = []
# Iterate over all image files in dataset directory
for filename in os.listdir(dataset_path):
    if filename.endswith("lr.jpg"):
        # Load image
        image = Image.open(os.path.join(dataset_path, filename))
        # Calculate SNR for image
        snr = calculate_snr(image)
        # Append SNR to list
        snr_values_lwir.append(snr)
    elif filename.endswith(".jpg"):
        # Load image
        image = Image.open(os.path.join(dataset_path, filename))
        # Calculate SNR for image
        snr = calculate_snr(image)
        # Append SNR to list
        snr_values_rgb.append(snr)

# Calculate average SNR for dataset
dataset_snr_lwir = np.mean(snr_values_lwir)
dataset_snr_rgb = np.mean(snr_values_rgb)
print("LWIR图像信噪比（SNR）：", dataset_snr_lwir)
print("RGB图像信噪比（SNR）：", dataset_snr_rgb)