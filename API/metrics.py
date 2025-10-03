import numpy as np
from skimage.metrics import structural_similarity as cal_ssim
import cv2

def MAE(pred, true):
    return np.mean(np.abs(pred-true)).sum()

def MSE(pred, true):
    return np.mean((pred-true)**2).sum()

def PSNR(pred, true):
    mse = MSE(pred,true)
    return 20. * np.log10(255. / np.sqrt(mse))

# def SSIM(pred, true, **kwargs):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2

#     img1 = pred.astype(np.float64)
#     img2 = true.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) 											* (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()


def metric(pred, true, mean, std, return_ssim_psnr=False, clip_range=[0, 1]):
    pred = pred*std + mean
    true = true*std + mean
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = np.sqrt(mse)
    if return_ssim_psnr:
        # pred = np.maximum(pred, clip_range[0])
        # pred = np.minimum(pred, clip_range[1])
        ssim, psnr = 0, 0
        for b in range(pred.shape[0]):
            ssim += cal_ssim(pred[b], true[b])
            psnr += PSNR(pred[b], true[b])
        ssim = ssim / (pred.shape[0])
        psnr = psnr / (pred.shape[0])
        return rmse, mae, ssim, psnr
    else:
        return rmse, mae