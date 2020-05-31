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
from aicsimageio.transforms import reshape_data
import numpy as np
import os
from tqdm import tqdm
import csv



def get_img_path_list(img_path_list, filepath):
  ''' Creates a list of image-path that will be used for loading the images later'''
  flist = os.listdir(filepath)
  flist.sort()
  for i in flist:
    img_slice_path = os.path.join(filepath, i)
    img_path_list.append(img_slice_path)
  return img_path_list

import csv
def append_txt_names(name, destination, filepath_list, k):
    os.chdir(destination)
    # list_number_of_files.append(i)  
    if k ==0:
        with open("log_names.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["slice_name","path_source"])
            writer.writerow([name, filepath_list[k]])
    else:
        with open("log_names.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, filepath_list[k]])
            

def create_slice_folder(root, folder_name, overwrite):
    counter = 0
    os.chdir(root)
    destination = os.path.join(root,folder_name)
    # remove folder if there was one before
    # create new folder
    if overwrite == True:
      if os.path.exists(destination)==True:
        shutil.rmtree(destination)
        os.mkdir(destination)
        os.chdir(destination)
      else:
        os.mkdir(destination)
        os.chdir(destination)
    if overwrite == False:
      if os.path.exists(destination)==True:
        check_destination = destination
        while os.path.exists(check_destination):
          check_destination = destination + "_"+"%s"%str(counter)
          counter+=1
        destination = check_destination
        os.mkdir(destination)
        os.chdir(destination)
      else:
        os.mkdir(destination)
        os.chdir(destination) 
    return destination


def run_slicer(filepath, folder_name, root, divisor, destination):
    # By running this the image will be saved in smaller slices in the decided folder #
    t=0
    img_path_list = []
    img_path_list = get_img_path_list(img_path_list, filepath)
    img = AICSImage(img_path_list[0])
    img = img.get_image_data("CYXZ", S=0, T=0)
    channels = img.shape[0]
    x_dim = img.shape[1]
    y_dim = img.shape[2]
    x_div = x_dim//divisor
    y_div = y_dim//divisor
    os.chdir(destination)
    for i in range(channels):
      if os.path.exists(str(i)):
        shutil.rmtree(str(i))
      os.mkdir(str(i))

    # k is for the different image
    for k in tqdm(range(len(img_path_list))):
        img_path = img_path_list[k]
        # to get the multiplyer that I can name the files correctly
        image_resolution = x_dim
        multiplyer = image_resolution/divisor
        for channel in range(channels):
          img = AICSImage(img_path_list[k])
          img = img.get_image_data("ZYX", S=0, T=0, C=channel)
          name_txt = ("%03d" %(t)+"-" +"%03d"  %(k) + "-"+"XXX")
          # if channel ==0:
          append_txt_names(name_txt,destination,img_path_list,k)
          for i in range(x_div):
            for j in range(y_div):
              img_crop = img
              # print(str((i*divisor))+":"+ str(((i+1)*divisor))+":" +","+ str((j*divisor))+":"+str(((j+1)*divisor))+":")
              img_crop = img_crop[:, (i*divisor):((i+1)*divisor):,(j*divisor):((j+1)*divisor)]
              # cv2_imshow(img_crop)
              print("%03d" %(t)+"-" +"%03d"  %(k) + "-"+"%02d" %((i*multiplyer)+j))
              name = ("%03d" %(t)+"-" +"%03d"  %(k) + "-"+"%02d" %((i*multiplyer)+j))
              #swap the axis to e able to save as tif file
              # img_crop = np.swapaxes(img_crop, 2, 0)
              img_crop = reshape_data(img_crop, "ZYX", "CZYX")
              img_crop= reshape_data(img_crop, "CZYX", "ZCYX")

              print("saving image {}".format(name))
              save_location = destination +"/"+ str(channel)
              os.chdir(save_location)
              with OmeTiffWriter("{}.tif".format(name)) as writer2:
                writer2.save(img_crop)