import numpy as np
import cv2

def calculate_snr(image):
    image = image.astype(np.float32)
    # 估计图像信号

    signal = np.mean(image/255)
    # 计算图像噪声
    noise = np.std(image/255)
    noise = np.mean((image/255 - np.mean(image/255)) ** 2)
    # 计算 SNR
    snr = 20 * np.log10(signal / noise)
    import pdb
    pdb.set_trace()
    return snr

# 读取图像
img=r'/home/yuanmaoxun/RSDet/utils/190006/vis_after_masked.png'
image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
# 计算 SNR
snr = calculate_snr(image)

print("图像信噪比（SNR）：", snr)
