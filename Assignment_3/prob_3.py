import itertools
import cv2
import numpy as np
from utils import add_white_gaussian_noise, mse, convert_img_uint8

import pdb

def get_high_pass_component(img):
    hp_filter = np.array([[-1,-1,-1],
                         [-1, 8, -1],
                         [-1, -1, -1]], dtype=np.float32)
    return cv2.filter2D(src=img, ddepth=-1, kernel=hp_filter)

def apply_sharpening_gain(img, hp_comp, gain):
    out_img = img + (gain - 1) * hp_comp
    return convert_img_uint8(out_img)

def gain_from_mse(orig, denoised, hp_denoised):
    gain = np.sum( (orig * hp_denoised) +  (hp_denoised * hp_denoised) -  (denoised * hp_denoised))/np.sum(hp_denoised * hp_denoised) 
    return gain

if __name__=="__main__":
    orig_img = cv2.imread('lighthouse2.bmp', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    denoised_img = cv2.imread('q2_part_3.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    hp_denoised_img = get_high_pass_component(denoised_img)
    cv2.imwrite('sharpened.png', hp_denoised_img)
    #Apply constant gains
    for gain in np.arange(1., 2.1, 0.1):
        sharpen_img = apply_sharpening_gain(denoised_img, hp_denoised_img, gain)
        cv2.imwrite('q3_constant_gain_{}.png'.format(gain*10), sharpen_img)
        print('Gain : {} MSE : {}'.format(gain, mse(orig_img.astype(np.uint8), sharpen_img)))

    hp_orig_img = get_high_pass_component(orig_img)
    mse_gain = gain_from_mse(orig_img, denoised_img, hp_denoised_img)
    print('MSE Gain is :', mse_gain)
    sharpen_img = apply_sharpening_gain(denoised_img, hp_denoised_img, mse_gain)
    cv2.imwrite('q3_mse_gain.png', sharpen_img)
    print('MSE : {}'.format( mse(orig_img.astype(np.uint8), sharpen_img)))