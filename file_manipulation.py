import shutil  # no need to import these, they're already imported at install
import csv
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import os

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

def run_manipulation():
    #Here we replace values in the old files
    #Change maximum pixel number
    replace("/content/gdrive/My Drive/pytorch_fnet/fnet/transforms.py",'n_max_pixels=9732096','n_max_pixels=20000000')
    replace("/content/gdrive/My Drive/pytorch_fnet/predict.py",'6000000','20000000')

    #Prevent resizing in the training and the prediction
    replace("/content/gdrive/My Drive/pytorch_fnet/predict.py","0.37241","1.0")
    replace("/content/gdrive/My Drive/pytorch_fnet/train_model.py","0.37241","1.0")

    #We add the necessary validation parameters here.
    f = open("/content/gdrive/My Drive/pytorch_fnet/scripts/train_model.sh", "r")
    contents = f.readlines()
    f.close()
    f = open("/content/gdrive/My Drive/pytorch_fnet/scripts/train_model.sh", "r")
    if not 'PATH_DATASET_VAL_CSV=' in f.read():
      contents.insert(10, 'PATH_DATASET_VAL_CSV="data/csvs/${DATASET}_val.csv"')
      contents.append('\n       --path_dataset_val_csv ${PATH_DATASET_VAL_CSV}')
    f.close()
    f = open("/content/gdrive/My Drive/pytorch_fnet/scripts/train_model.sh", "w")
    contents = "".join(contents)
    f.write(contents)
    f.close()


    #Clear the White space from train.sh

    with open('/content/gdrive/My Drive/pytorch_fnet/scripts/train_model.sh', 'r') as inFile,\
        open('/content/gdrive/My Drive/pytorch_fnet/scripts/train_model_temp.sh', 'w') as outFile:
        for line in inFile:
            if line.strip():
                outFile.write(line)
    os.remove('/content/gdrive/My Drive/pytorch_fnet/scripts/train_model.sh')
    os.rename('/content/gdrive/My Drive/pytorch_fnet/scripts/train_model_temp.sh','/content/gdrive/My Drive/pytorch_fnet/scripts/train_model.sh')

    #Datasets

    #Change checkpoints
    replace("/content/gdrive/My Drive/pytorch_fnet/train_model.py","'--interval_save', type=int, default=500","'--interval_save', type=int, default=100")

    #Adapt Class Dataset for Tiff files
    replace("/content/gdrive/My Drive/pytorch_fnet/train_model.py","'--class_dataset', default='CziDataset'","'--class_dataset', default='TiffDataset'")
