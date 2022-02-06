import os
import numpy as np
import cv2

import pdb

class SIFT:
    """Implementation of SIFT
    """
    def __init__(self, sigma=2.0, num_scales_in_octave=4):
        """Concstructor.

        Args:
            sigma (float, optional): Sigma value for gaussian. Defaults to 2.0.
            num_scales_in_octave (int, optional): Number of scales in an octave. Defaults to 4.
        """
        self.sigma = sigma
        self.num_scales_in_octave = num_scales_in_octave
        
    def genearate_base_image(self, image):
        """Generates base image in an octave

        Args:
            image (np.array): input image

        Returns:
            [np.array: gaussian blurred image
        """
        return cv2.GaussianBlur(image, (0,0), sigmaX=self.sigma, sigmaY=self.sigma)

    def generate_gaussian_images(self, image, base_image):
        """Generates gaussian images. 

        Args:
            image (np.array): input image
            base_image (np.array): base image in an octave

        Returns:
            [list]: gaussian images in an octave
        """
        octave_images = [base_image]
        k = 2**(1./self.num_scales_in_octave)
        #cv2.imwrite('gauss_image_0.jpg', octave_images[-1].astype(np.uint8))
        for i in range(1,self.num_scales_in_octave+1):
            scale_space_sigma = (k**i) * self.sigma
            #print(scale_space_sigma)
            octave_images.append(cv2.GaussianBlur(image, (0,0), sigmaX=scale_space_sigma, sigmaY=scale_space_sigma) )
            #cv2.imwrite('gauss_image_'+str(i)+'.jpg', octave_images[-1].astype(np.uint8))
        return octave_images

    def generate_difference_of_gaussians(self, gaussian_octave_images):
        """Generates difference of gaussian images. 

        Args:
            gaussian_octave_images (list): gaussian images in an octave

        Returns:
            [list]: difference of gaussian images in an octave
        """
        octave_dog = []
        for i in range(1,len(gaussian_octave_images)):
            octave_dog.append(gaussian_octave_images[i] - gaussian_octave_images[i-1])
        return octave_dog

    def isPointExtrema(self, prev_scale_space, curr_scale_space, next_scale_space, x, y):
        """Returns true if point is extreme in 3x3x3 volume in an octave

        Args:
            prev_scale_space (np.array): previous scale in an octave
            curr_scale_space (np.array): current scale in an octave
            next_scale_space (np.array): next scale in an octave
            x (int): location x in image
            y (int): location y in image
        Returns:
            [boolean]: True if point is extremum.
        """
        center_val = curr_scale_space[y,x]
        neighbourhood = np.zeros((3,3,3), dtype=np.float32)
        neighbourhood[0,:,:] = prev_scale_space[y-1:y+2, x-1:x+2]
        neighbourhood[1,:,:] = curr_scale_space[y-1:y+2, x-1:x+2]
        neighbourhood[2,:,:] = next_scale_space[y-1:y+2, x-1:x+2]

        #pdb.set_trace()
        if center_val == np.max(neighbourhood):
            return True
        elif center_val == np.min(neighbourhood):
            return True
        else:
            return False

    def scale_space_extrema_detection(self, image, base_image):
        """Scale Space extrema detection

        Args:
            image (np.array): input image
            base_image (np.array): base image in an octave

        Returns:
            [set]: detected keypoints
        """
        #pdb.set_trace()
        octave_images = self.generate_gaussian_images(image, base_image)
        octave_dog = self.generate_difference_of_gaussians(octave_images)

        keypoints_set = set()
        for octave_index in range(1, len(octave_dog)-1):
            #check across 3 x 3 window
            for y in range(1, image.shape[0]-1):
                for x in range(1, image.shape[1]-1):
                    if self.isPointExtrema(octave_dog[octave_index-1], octave_dog[octave_index], octave_dog[octave_index+1], x, y):
                        keypoints_set.add((x,y))
        return keypoints_set

    def estimate_key_points(self, image):
        """Estimates keypoints.

        Args:
            image (np.array): input image

        Returns:
            [set]: detected keypoints
        """
        image = image.astype(np.float32)
        base_image = self.genearate_base_image(image)
        keypoints = self.scale_space_extrema_detection(image, base_image)
        return keypoints

def draw_keypoints(img, keypoints):
    """Draw keypoints in an image.

    Args:
        img (np.array): input image
        keypoints (set): input keypoints

    Returns:
        [np.array]: image with keypoints
    """
    #pdb.set_trace()
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
        img = cv2.circle(img, kp, 2, (0,255,0), -1)
    return img

def add_noise(img):
    """Adds noise to an image.

    Args:
        img (np.array): input image

    Returns:
        [np.array]: noisy image
    """
    gaussian = np.random.normal(0, 10, (img.shape[0],img.shape[1]))
    noisy_image = np.zeros(img.shape, np.float32)
    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img[:, :, 2] + gaussian
    return noisy_image.astype(np.uint8)

def save_results(image_path, img_id):
    """Save results for assignment.

    Args:
        image_path (str): path of an input image.
        img_id (str): id for outputs
    """
    sift_feature_extractor = SIFT()
    
    img = cv2.imread(image_path)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = sift_feature_extractor.estimate_key_points(grayscale_img)
    print('length of keypoints (normal)', len(keypoints))
    cv2.imwrite(img_id + '_keypoints_normal.jpg', draw_keypoints(img,keypoints))

    #rotate
    rot_img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = sift_feature_extractor.estimate_key_points(grayscale_img)
    print('length of keypoints (rot)', len(keypoints))
    cv2.imwrite(img_id + '_keypoints_rotate.jpg', draw_keypoints(rot_img,keypoints))

    #noise
    noisy_img = add_noise(img)
    grayscale_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
    keypoints = sift_feature_extractor.estimate_key_points(grayscale_img)
    print('length of keypoints (noise)', len(keypoints))
    cv2.imwrite(img_id + '_keypoints_noisy.jpg', draw_keypoints(noisy_img,keypoints))

    #blur
    blur_img = cv2.GaussianBlur(img, (5,5), sigmaX=0, sigmaY=0)
    grayscale_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    keypoints = sift_feature_extractor.estimate_key_points(grayscale_img)
    print('length of keypoints (blur)', len(keypoints))
    cv2.imwrite(img_id + '_keypoints_blur.jpg', draw_keypoints(blur_img,keypoints))

    #downsampling
    rsz_img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    grayscale_img = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY)
    keypoints = sift_feature_extractor.estimate_key_points(grayscale_img)
    print('length of keypoints (resize)', len(keypoints))
    cv2.imwrite(img_id + '_keypoints_resized.jpg', draw_keypoints(rsz_img,keypoints))

if __name__ == '__main__':
    sift_feature_extractor = SIFT()
    save_results('baboon.jpg', 'baboon')
    save_results('rubberwhale1.png', 'rubberwhale1')
    

