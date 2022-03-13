import itertools
import cv2
import numpy as np
from utils import add_white_gaussian_noise, mse
import pdb

np.random.seed(4)
def run_part_1(original_img, noisy_img):
    filter_length = [3, 7, 11]
    filter_sigma = [0.1, 1, 2, 4, 8]
    for item in list(itertools.product(filter_length, filter_sigma)):
        print('Length : {} \t Sigma : {}'.format(item[0], item[1]))
        low_freq_denoised = cv2.GaussianBlur(noisy_img,(item[0],item[0]),item[1])
        cv2.imwrite('q2_part_1_denoised_length_{}_sigma_{}.png'.format(item[0],item[1]), low_freq_denoised)
        print('MSE is : {}\n'.format(mse(original_img, low_freq_denoised)))

def mmse_estimate(noisy_img):
    #use best setting
    n = 3
    std_dev = 1. 
    mu_y = cv2.GaussianBlur(noisy_img,(n,n), std_dev)
    y1 = noisy_img - mu_y
    mean_y1 = np.sum(y1)/(y1.shape[0] * y1.shape[1])
    var_y1 = np.sum( (y1 - mean_y1)**2)/(y1.shape[0] * y1.shape[1])
    weights = cv2.getGaussianKernel(n, 1.)
    weights = weights * weights.transpose(1, 0)
    w00 = weights[int(n/2), int(n/2)]
    var_z1 = ( (1. - w00)**2 + (np.sum(weights**2) - (w00**2)) ) * 100
    var_x1 = var_y1 - var_z1
    mmse_x = mu_y + ( (var_x1)/(var_x1 + var_z1) ) * y1
    return mmse_x

def run_part_2(original_img, noisy_img):
    mmse_x =  mmse_estimate( noisy_img)
    cv2.imwrite('q2_part_2.png', mmse_x)
    print('MSE is : {}'.format(mse(original_img, mmse_x)))

def adaptive_mmse(original_img, noisy_img, patch_size=11, step=6):
    acc_mmse_x = np.zeros(original_img.shape, dtype=np.float32)
    acc_denom = np.zeros(original_img.shape, dtype=np.int32)
    for y in range(0,original_img.shape[0],step):
        for x in range(0, original_img.shape[1],step):
            x_end = min(x+patch_size, original_img.shape[1])
            y_end = min(y+patch_size, original_img.shape[0])
            acc_mmse_x[y:y_end, x:x_end] += mmse_estimate(noisy_img[y:y_end, x:x_end])
            acc_denom[y:y_end, x:x_end] += 1
    return np.divide(acc_mmse_x, acc_denom)

def run_part_3(original_img, noisy_img):
    mmse_x =  adaptive_mmse(original_img, noisy_img)
    cv2.imwrite('q2_part_3.png', mmse_x)
    print('MSE is : {}'.format(mse(original_img, mmse_x)))

if __name__=="__main__":
    img = cv2.imread('lighthouse2.bmp', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('q2_gt_img.png', img)
    noisy_img = add_white_gaussian_noise(img)
    cv2.imwrite('q2_noisy_img.png', noisy_img)
    run_part_1(img, noisy_img)
    run_part_2(img, noisy_img)
    run_part_3(img, noisy_img)
