import os
import numpy as np
import scipy.io
import cv2
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
from scipy import stats
import matplotlib.pyplot as plt

def find_max_and_min(inp_list):
    minpos = inp_list.index(min(inp_list))
    maxpos = inp_list.index(max(inp_list))
    return minpos, maxpos

def create_scatter_plot(x, y1, y2, y3):
    p1 = plt.scatter(x, y1, c='r', marker='o', label='MSE' )
    plt.legend()
    plt.title('HOS vs MSE')
    plt.xlabel('HOS score')
    plt.ylabel('MSE score')
    plt.savefig('mse.png')
    plt.close()

    p2 = plt.scatter(x, y2, c='b', marker='o', label='SSIM' )
    plt.legend()
    plt.title('HOS vs SSIM')
    plt.xlabel('HOS score')
    plt.ylabel('SSIM score')
    plt.savefig('ssim.png')
    plt.close()

    p3 = plt.scatter(x, y3, c='g', marker='o', label='LPIPS' )
    plt.legend()
    plt.title('HOS vs LPIPS')
    plt.xlabel('HOS score')
    plt.ylabel('LPIPS score')
    plt.savefig('lpips.png')
    plt.close()

def save_pairs(blur_img, ref_img, save_img_name):
    blur = cv2.imread(blur_img)
    ref = cv2.imread(ref_img)
    final_img = np.concatenate((blur, ref), axis=1)
    cv2.imwrite(save_img_name, final_img)

#dict_keys(['__header__', '__version__', '__globals__', 'blur_dmos', 'blur_orgs', 'refnames_blur'])
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization  
def mse(ref, img):
    float_ref = (ref.copy()).astype(np.float64)
    float_img = (img.copy()).astype(np.float64)
    denom = 1
    for elem in ref.shape:
        denom = denom * elem
    return np.sum((float_ref - float_img)**2)/(denom)

def get_ssim(ref, img):
    if len(img.shape) == 2:
        return ssim(ref, img, data_range=img.max() - img.min())
    else:
        return ssim(ref, img, data_range=img.max() - img.min(), channel_axis=2)

def get_lpips(ref, img):
    img_0 = torch.from_numpy(np.transpose(ref[:,:, ::-1].copy(), (2,0,1))).unsqueeze(0)
    img_1 = torch.from_numpy(np.transpose(img[:,:, ::-1].copy(), (2,0,1))).unsqueeze(0)
    return loss_fn_vgg.forward(img_0, img_1).detach().numpy()[0,0,0,0]

data_dir = './hw5'
info_mat = scipy.io.loadmat( os.path.join(data_dir, 'hw5.mat') )
blur_images_dir = os.path.join(data_dir, 'gblur')
ref_images_dir = os.path.join(data_dir, 'refimgs')
save_dir = './prob_3'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

metric_mse = []
metric_ssim = []
metric_lpips = []
metric_dmos = []
for i in range(info_mat['blur_dmos'].shape[1]):
    filename = 'img{}.bmp'.format(i+1)
    filepath = os.path.join(blur_images_dir, filename)
    if not os.path.exists(filepath) or info_mat['blur_orgs'][0,i] == 1:
        continue
    img = cv2.imread(filepath)
    ref = cv2.imread(os.path.join(ref_images_dir, str(info_mat['refnames_blur'][0,i][0])) )
    
    metric_mse.append(mse(ref, img))
    metric_ssim.append(get_ssim(ref, img))
    metric_lpips.append(get_lpips(ref, img))
    metric_dmos.append(info_mat['blur_dmos'][0,i])
    print('filname : {} \t DMOS : {} \t MSE : {} \t SSIM : {} \t LPIPS : {}'.format(filename \
        , metric_dmos[-1], metric_mse[-1], metric_ssim[-1], metric_lpips[-1]))
    #print(info_mat['blur_dmos'][0,i], info_mat['blur_orgs'][0,i])
#pdb.set_trace()

print('Human and MSE')
print(stats.spearmanr(metric_dmos, metric_mse))

print('Human and SSIM')
print(stats.spearmanr(metric_dmos, metric_ssim))

print('Human and LPIPS')
print(stats.spearmanr(metric_dmos, metric_lpips))

min_index, max_index = find_max_and_min(metric_mse)
save_pairs(os.path.join(blur_images_dir, 'img{}.bmp'.format(min_index+1)), 
            os.path.join(ref_images_dir, str(info_mat['refnames_blur'][0,min_index][0])),
            os.path.join(save_dir, 'mse_min_pair.png'))
save_pairs(os.path.join(blur_images_dir, 'img{}.bmp'.format(max_index+1)), \
    os.path.join(ref_images_dir, str(info_mat['refnames_blur'][0,max_index][0])),\
    os.path.join(save_dir, 'mse_max_pair.png'))
    
print('Minimum ans Maximum MSE is for {} & {} for images {} & {}'.format(metric_mse[min_index], \
    metric_mse[max_index], min_index, max_index))

min_index, max_index = find_max_and_min(metric_ssim)
save_pairs(os.path.join(blur_images_dir, 'img{}.bmp'.format(min_index+1)), 
            os.path.join(ref_images_dir, str(info_mat['refnames_blur'][0,min_index][0])),
            os.path.join(save_dir, 'ssim_min_pair.png'))
save_pairs(os.path.join(blur_images_dir, 'img{}.bmp'.format(max_index+1)), \
    os.path.join(ref_images_dir, str(info_mat['refnames_blur'][0,max_index][0])),\
    os.path.join(save_dir, 'ssim_max_pair.png'))
print('Minimum ans Maximum SSIM is for {} & {} for images {} & {}'.format(metric_ssim[min_index], \
    metric_ssim[max_index], min_index, max_index))

min_index, max_index = find_max_and_min(metric_lpips)
save_pairs(os.path.join(blur_images_dir, 'img{}.bmp'.format(min_index+1)), 
            os.path.join(ref_images_dir, str(info_mat['refnames_blur'][0,min_index][0])),
            os.path.join(save_dir, 'lpips_min_pair.png'))
save_pairs(os.path.join(blur_images_dir, 'img{}.bmp'.format(max_index+1)), \
    os.path.join(ref_images_dir, str(info_mat['refnames_blur'][0,max_index][0])),\
    os.path.join(save_dir, 'lpips_max_pair.png'))
print('Minimum ans Maximum LPIPS is for {} & {} for images {} & {}'.format(metric_lpips[min_index], \
    metric_lpips[max_index], min_index, max_index))

create_scatter_plot(metric_dmos, metric_mse, metric_ssim, metric_lpips)