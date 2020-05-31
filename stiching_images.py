import numpy
import os
import shutil
import random
from IPython.display import clear_output
from google.colab.patches import cv2_imshow
from aicsimageprocessing import resize
from aicsimageio import AICSImage
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from aicsimageio import AICSImage, imread
from aicsimageio.writers import png_writer 
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
import numpy as np
import os
from tqdm import tqdm

def get_img_path_list(img_path_list, filepath):
  ''' Creates a list of image-path that will be used for loading the images later'''
  flist = os.listdir(filepath)
  flist.sort()
  for i in flist:
    img_slice_path = os.path.join(filepath, i)
    # if "Prediction" in img_slice_path:
    img_path_list.append(img_slice_path)
  return img_path_list


def load_img(img_path, output_dim):
    '''load the image, selects the right dimensions and returns the image in 
    form eg 16x512x512 the dimensions of the input file and the multiplyer 
    needed for the assignment of the image to the right position'''
    img = AICSImage(img_path)
    img = img.get_image_data("ZYX", S=0, C=0, T=0)
    x_dim = img.shape[1]
    z_dim = img.shape[0]

    multiplyer = output_dim//x_dim
    # img = np.swapaxes(img, 2, 0)
    return img, multiplyer, x_dim, z_dim

def create_folder(root, folder_name):
    os.chdir(root)
    destination = os.path.join(root,folder_name)
    # remove folder if there was one before
    # create new folder
    if os.path.exists(destination):
      shutil.rmtree(destination)
      os.mkdir(destination)
    else:
      os.mkdir(destination)
    os.chdir(destination)
    return destination

def run_stitching(destination, filepath, output_dim):
    os.chdir(destination)
    img_path_list = []
    img_path_list = get_img_path_list(img_path_list, filepath)
    img, multiplyer, input_dim, z_dim= load_img(img_path_list[0], output_dim)
    # in case if i want to do a time series conversion - another loop
    t=0
    for k in range(len(img_path_list)//(multiplyer*multiplyer)):
      zeros = np.zeros((z_dim, output_dim, output_dim))

      for i in tqdm(range(multiplyer)):
        for j in range(multiplyer):
          counter_img_path = j+i*multiplyer
          # print(counter_img_path)
          img_path = img_path_list[counter_img_path]
          img, multiplyer,input_dim, z_dim = load_img(img_path, output_dim)
          zeros[:,i*input_dim:(i+1)*input_dim,j*input_dim:(j+1)*input_dim] = img
          # print(img.shape, zeros[:,i*input_dim:(i+1)*input_dim,j*input_dim:(j+1)*input_dim].shape)
          # print([":," + str(i*input_dim) +":"+str((i+1)*input_dim) + "," + str(j*input_dim)+":"+str((j+1)*input_dim)])
      # print(zeros.shape)
      name = ("%03d" %(t)+"-"+"%03d"  %(k) )
      print(name)
      with OmeTiffWriter("{}.tif".format(name)) as writer2:
        writer2.save(zeros)

def rename_files(filepath, destination):
    # provide folder with name
    master_folder = os.path.dirname(filepath)
    name_txt_file = master_folder + "/log_names.csv"

    dataframe = pd.read_csv(name_txt_file)
    path_names = dataframe["path_source"]
    flist = os.listdir(destination)
    for stiched_name, path in zip(flist, path_names):
      original_name = os.path.basename(path)
      old = destination + "/"+ stiched_name
      new = destination + "/R_" + original_name
      os.rename(old, new)

