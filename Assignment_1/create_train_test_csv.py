import os
import yaml

import pdb

def get_img_paths(folder, id):
    """Get Image Paths

    Args:
        folder (str): folder containing images
        id (int): label of the folder

    Returns:
        [tuple]: Image paths, Image Labels
    """
    img_paths = []
    img_label = []
    for file_ in os.listdir(folder):
        if os.path.splitext(file_)[1] in ['.jpg']:
            img_paths.append(os.path.join(folder,file_))
            img_label.append(id)
    #print(img_paths, img_label)
    return img_paths, img_label

def write_csv_file(file_name, images, labels ):
    """Writes csv file

    Args:
        file_name (str): name of the output file
        images (list): list of the input images
        labels (list): list of the input labels
    """
    file_= open(file_name,'w')
    file_.write('Image Paths,Label')
    file_.write('\n')
    for i in range(len(images)):
        file_.write('{},{}\n'.format(images[i], labels[i]))
    file_.close()

#read yaml file
with open('config.yaml','r') as file:
    cfg= yaml.safe_load(file)

categories_to_id = cfg['categories']

#Create Training Data
img_paths = []
img_labels = []
for category in categories_to_id.keys():
    curr_img_paths, curr_img_labels = get_img_paths(os.path.join(cfg['data_folder'],category+"_"+ 'train'), categories_to_id[category])
    img_paths = img_paths + curr_img_paths
    img_labels = img_labels + curr_img_labels
write_csv_file('train.csv', img_paths, img_labels)

#Create Test Data
img_paths = []
img_labels = []
for category in categories_to_id.keys():
    curr_img_paths, curr_img_labels = get_img_paths(os.path.join(cfg['data_folder'],category+"_"+ 'test'), categories_to_id[category])
    img_paths = img_paths + curr_img_paths
    img_labels = img_labels + curr_img_labels
write_csv_file('test.csv', img_paths, img_labels)

