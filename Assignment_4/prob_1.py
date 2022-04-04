import os
import numpy as np
import cv2
import math
from bitstring import BitArray

import pdb 

# def mse(a, b):
#     float_a = (a.copy()).astype(np.float64)
#     float_b = (b.copy()).astype(np.float64)
#     return np.sum((float_a - float_b)**2)/(float_a.shape[0] * float_a.shape[1])

def mse(img1, img2):

    delta = img1.astype(np.float32)-img2.astype(np.float32)

    accum = np.multiply(delta, delta)

    mse_accum = np.sum(accum) / (img1.shape[0]*img1.shape[1])

    return mse_accum

class JPEGCommpression:
    def __init__(self, a=10, b=40, c=20):
        self.id = 'jpeg_comprssion'
        self.q_matrix = b * np.ones((8,8), dtype=np.int32)
        self.q_matrix[0,0] = c
        self.q_matrix[0,1] = a
        self.q_matrix[1,0] = a
        self.q_matrix[1,1] = a
        
    def encode_value(self, a):
        if a == 0:
            return '0'
        elif a == 1:
            return '101'
        elif a == -1:
            return '100'
        else:
            num_ones = int(math.log(abs(a), 2)) + 1
            head = '1'*num_ones + '0'
            bin_str = bin(abs(a))[2:]
            if a < 0:
                bin_str = ''.join(['1' if ch == '0' else '0' for ch in bin_str]) #reverse
            return head + bin_str

    def encode(self, quantized_dct):
        encoded_stream = ''
        for y in range(8):
            for x in range(8):
                encoded_stream = encoded_stream + self.encode_value(quantized_dct[y,x])
        #pdb.set_trace()
        return encoded_stream

    def save_file(self, jpeg_stream, save_path):
        compressed_file = open(save_path, 'wb')
        b = BitArray(bin=jpeg_stream)
        b.tofile(compressed_file)
        compressed_file.close()

    def compress(self, img, save_path):
        print(self.q_matrix)
        #pdb.set_trace()
        jpeg_stream = ''
        for y in range(0,img.shape[0],8):
            for x in range(0, img.shape[1], 8):
                block = img[y:y+8, x:x+8].astype(np.float64)
                block_dct = cv2.dct(block)
                quantized_block_dct = (np.divide(block_dct, self.q_matrix) + 0.5).astype(np.int32) 
                jpeg_stream = jpeg_stream + self.encode(quantized_block_dct)
        print('Size of image in bits : ', 256 * 256 * 8)
        print('Size of compressed file in bits', len(jpeg_stream))
        print('Compression Ratio : ', float(256 * 256 * 8)/len(jpeg_stream))
        self.save_file(jpeg_stream, save_path)
        return jpeg_stream
        #print(int(jpeg_stream,2))
    
    def decode_val(self, jpeg_stream, id):
        if jpeg_stream[id] == '0':
            return 1,0
        
        #decoding of other values
        num_ones = 0
        ctr = id
        while jpeg_stream[ctr] != '0':
            num_ones += 1
            ctr += 1
        ctr += 1
        bin_str = ''
        for i in range(num_ones):
            bin_str = bin_str + jpeg_stream[ctr+i]
        is_neg = False
        if jpeg_stream[ctr] == '0': #negative
            is_neg = True
            bin_str = ''.join(['1' if ch == '0' else '0' for ch in bin_str]) #reverse
        #pdb.set_trace()
        value = int('0b'+bin_str,2)
        return 2*num_ones+1, value if not is_neg else -value


    def decode_and_eval(self, original_img, jpeg_stream, save_path):
        patch_id = 0
        id = 0
        decoded_values = []
        decoded_img = np.zeros((256,256), dtype=np.uint8)
        while id < len(jpeg_stream):
            #print(id)
            read_indices, value = self.decode_val(jpeg_stream, id)
            decoded_values.append(value)
            id = id + read_indices
            if len(decoded_values) == 64:
                patch_y = int(patch_id/32) * 8
                patch_x = (patch_id % 32) * 8
                block = np.reshape(np.array(decoded_values, dtype=np.int32), (8,8) )
                block = np.multiply(block, self.q_matrix).astype(np.float64)
                inv_dct = cv2.idct(block)
                inv_dct[inv_dct > 255] = 255
                inv_dct[inv_dct < 0] = 0
                #pdb.set_trace()
                decoded_img[patch_y:patch_y+8, patch_x:patch_x+8] = inv_dct.astype(np.uint8)
                patch_id += 1
                decoded_values = [] 

        cv2.imwrite(save_path, decoded_img)
        mse_metric = mse(original_img, decoded_img)
        print('Reconstruction Error (MSE):', mse_metric)
        return mse_metric

if __name__=="__main__":
    img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
    save_dir = './q1'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    #part a
    print('Part A a=10, b=40, c=20')
    jpeg_compressor_a = JPEGCommpression()

    #test encoder
    # print('0 : ', jpeg_compressor_a.encode_value(0))
    # for i in range(1,16):
    #     print(i,':',jpeg_compressor_a.encode_value(i))
    #     print(-i,':',jpeg_compressor_a.encode_value(-i))
    
    encoded_stream = jpeg_compressor_a.compress(img, os.path.join(save_dir, 'part_a_compressed.bin'))
    jpeg_compressor_a.decode_and_eval(img, encoded_stream,  os.path.join(save_dir, 'part_a_reconstructed.png'))
    part_a_compressed_len = len(encoded_stream)
    print('=' * 100 + '\n')
    exit()
    #Part B

    #part b
    print('Part B a=1, b=1, c=1')
    jpeg_compressor_b = JPEGCommpression(a=1, b=1, c=1)
    encoded_stream = jpeg_compressor_b.compress(img, os.path.join(save_dir, 'part_b_compressed.bin'))
    jpeg_compressor_b.decode_and_eval(img, encoded_stream,  os.path.join(save_dir, 'part_b_reconstructed.png'))
    print('=' * 100 + '\n')

    
    #Part c
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    min_values = (10,10,10)
    min_mse = float(1e+100)
    #part c
    for a in values:
        for b in values:
            for c in values:
                print('a={}, b={}, c={}'.format(a,b,c))
                jpeg_compressor_b = JPEGCommpression(a=a, b=b, c=c)
                file_name = 'part_c_id_a_{}_b_{}_c_{}_'.format(a,b,c)
                encoded_stream = jpeg_compressor_b.compress(img, os.path.join(save_dir, file_name + 'compressed.bin'))
                mse_val = jpeg_compressor_b.decode_and_eval(img, encoded_stream,  os.path.join(save_dir, file_name +'reconstructed.png'))
                if len(encoded_stream) < part_a_compressed_len:
                    if mse_val < min_mse:
                        min_mse = mse_val
                        min_values = (a,b,c)
                print('-' * 100, '\n')

    print('best values of a,b and c : ' , min_values)