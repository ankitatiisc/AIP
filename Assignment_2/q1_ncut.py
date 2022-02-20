import cv2
import numpy as np
import math

from scipy.sparse.linalg import eigsh
from matplotlib import pyplot as plt
import pdb

def get_value_from_gaussian(x, sigma):
    return math.exp( -(x**2)/ (sigma**2))

def get_eucledian_distance(a,b):
    return math.sqrt(a**2 + b**2)

class NCut:
    def __init__(self, radius=10):
        print('NCUT Algorithm')
        self.sigma_spatial = 10
        self.sigma_intensity = 120
        self.dist_threshold = radius
        self.spatial_lut = self.get_spatial_lut(self.dist_threshold)
        self.intensity_lut = self.get_intensity_lut()
        
    def get_spatial_lut(self, thresh):
        lut = np.zeros((thresh+1,thresh+1), dtype=np.float64)
        for i in range(thresh+1):
            for j in range(thresh+1):
                lut[i][j] = get_value_from_gaussian(math.sqrt(i*i+j*j), self.sigma_spatial)
        #pdb.set_trace()
        return lut

    def get_intensity_lut(self):
        lut = np.zeros(256, dtype=np.float64)
        for i in range(256):
            lut[i] = get_value_from_gaussian(i, self.sigma_intensity)
        #pdb.set_trace()
        return lut

    def create_graph(self, img):
        # number of nodes : H * W
        N = img.shape[0] * img.shape[1]
        adj_matrix = np.zeros((N, N), dtype=np.float64)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                x_range_min = max(0, x-self.dist_threshold)
                y_range_min = max(0, y-self.dist_threshold)
                x_range_max = min(img.shape[1]-1, x+self.dist_threshold)
                y_range_max = min(img.shape[0]-1, y+self.dist_threshold)    
                for i in range(y_range_min,y_range_max+1): 
                    for j in range(x_range_min,x_range_max+1):
                        #pdb.set_trace()
                        val_intensity = self.intensity_lut[abs(img[y,x] - img[i,j])]
                        val_dist = self.spatial_lut[abs(y-i)][abs(x-j)]
                        #val_intensity = get_value_from_gaussian(img[y,x] - img[i,j], self.sigma_intensity)
                        #val_dist = get_value_from_gaussian(get_eucledian_distance(y-i,x-j), self.sigma_spatial)
                        adj_matrix[y*img.shape[1]+x, i*img.shape[1]+j] = val_intensity * val_dist
        return adj_matrix

    def discretize(self, vec, shape):
        discrete_mask = np.zeros(vec.shape, np.float32)
        continuous_mask = ((vec - vec.min())/(vec.max()-vec.min()))*255
        discrete_mask[vec > 0] = 255
        continuous_mask = continuous_mask.astype(np.uint8)
        continuous_mask = continuous_mask.reshape(shape)
        discrete_mask = discrete_mask.astype(np.uint8)
        discrete_mask = discrete_mask.reshape(shape)
        colored_disc_mask = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        colored_disc_mask[:,:] = [0,255,0]
        colored_disc_mask[discrete_mask > 0] = [255,0,0]
        #mask[vec > 0] = 255
        return colored_disc_mask, continuous_mask

    def process(self, img):
        img = img.astype(np.int32)
        N = img.shape[0] * img.shape[1]
        W = self.create_graph(img)
        W_img = (W.copy() / W.max()) * 255.0

        W_img = cv2.resize(W_img, (512, 512))

        cv2.imwrite('./figures/graph.png', W_img)
        D = np.zeros((N,N), dtype=np.float64)
        for i in range(N):
            D[i,i] = math.sqrt(W[i,:].sum())
        D_inv = np.linalg.inv(D)
        
        M = np.matmul(D_inv , np.matmul((D - W) , D_inv) )  

        evals, evecs = eigsh(M, k=5, which='LM')
        
        for i in range(5):
            cut_vec = evecs[:,i]
            cut_vec = cut_vec.reshape((N,1))
            discrete_mask, continous_mask = self.discretize(cut_vec, (img.shape[0], img.shape[1]))
            color_img = cv2.applyColorMap(continous_mask, cv2.COLORMAP_HOT)
            cv2.imwrite('cont_output_'+str(5-i) + '.jpg', color_img)

            cv2.imwrite('disc_output_'+str(5-i) + '.jpg', discrete_mask)
            
            plt.hist(cut_vec, bins = 100)
            plt.savefig('./figures/hist4_' + str(i) + '.png')
            plt.clf()

if __name__ == '__main__':
   
    img = np.zeros((100,100), dtype=np.uint8)
    img = cv2.circle(img, (50,50), 30, 255, -1)
    #img[30:70,30:70] = 255
    #img = cv2.imread('horse.jpg', cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, (100,100))
    cv2.imwrite('q1_input.jpg', img)
    ncut = NCut (radius=img.shape[0]//4)
    ncut.process(img)
