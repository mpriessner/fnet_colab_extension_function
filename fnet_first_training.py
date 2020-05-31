
import random
import os
import csv
import shutil
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

#This function replaces the old default files with new values
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


def perform_replacements():
    #Here we replace values in the old files
    #Change maximum pixel number
    replace("/content/gdrive/My Drive/pytorch_fnet/fnet/transforms.py",'n_max_pixels=9732096','n_max_pixels=20000000')
    replace("/content/gdrive/My Drive/pytorch_fnet/predict.py",'6000000','20000000')

    #Prevent resizing in the training and the prediction
    replace("/content/gdrive/My Drive/pytorch_fnet/predict.py","0.37241","1.0")
    replace("/content/gdrive/My Drive/pytorch_fnet/train_model.py","0.37241","1.0")

    #Datasets

    #Change checkpoints
    replace("/content/gdrive/My Drive/pytorch_fnet/train_model.py","'--interval_save', type=int, default=500","'--interval_save', type=int, default=100")

    #Adapt Class Dataset for Tiff files
    replace("/content/gdrive/My Drive/pytorch_fnet/train_model.py","'--class_dataset', default='CziDataset'","'--class_dataset', default='TiffDataset'")


def create_folders_for_training(fnet_dataset_path, Training_source, Training_target, fnet_data, source_name, target_name, model_path_name, model_name, continue_training=False, Pretrained_model_folder="", replace_existing_dataset=True):
    if not os.path.exists(fnet_dataset_path):
      #shutil.copytree(own_dataset,'/content/gdrive/My Drive/pytorch_fnet/data/'+dataset)
      os.makedirs(fnet_dataset_path)
      shutil.copytree(Training_source, fnet_dataset_path +'/'+source_name)
      shutil.copytree(Training_target, fnet_dataset_path +'/'+target_name)
      os.mkdir(fnet_dataset_path+'/Validation_Input')
      os.mkdir(fnet_dataset_path+'/Validation_Target')

    else: 
      if replace_existing_dataset == True:
        shutil.rmtree(fnet_dataset_path)
        os.makedirs(fnet_dataset_path)
        shutil.copytree(Training_source, fnet_dataset_path +'/'+source_name)
        shutil.copytree(Training_target, fnet_dataset_path +'/'+target_name)
        os.mkdir(fnet_dataset_path+'/Validation_Input')
        os.mkdir(fnet_dataset_path+'/Validation_Target')
      else:
        print("make sure that the data you want to process is in the following folder: /content/gdrive/My Drive/pytorch_fnet/data/ + dataset-name")

    if continue_training == False:
        #create model-path-folder as selected by the user on g-drive
        if os.path.exists(model_path_name):
          shutil.rmtree(model_path_name)
          os.mkdir(model_path_name)
        else:
          os.mkdir(model_path_name)
          #create model-path-folder in saved_models for folders in pytorch_fnet
        if os.path.exists("/content/gdrive/My Drive/pytorch_fnet/saved_models" + "/" + model_name):
          shutil.rmtree("/content/gdrive/My Drive/pytorch_fnet/saved_models" + "/" + model_name)
          os.mkdir("/content/gdrive/My Drive/pytorch_fnet/saved_models" + "/" + model_name)
        else:
          os.mkdir("/content/gdrive/My Drive/pytorch_fnet/saved_models" + "/" + model_name)
    elif continue_training == True:
        Pretrained_model_name = os.path.basename(Pretrained_model_folder)
        if os.path.exists('/content/gdrive/My Drive/pytorch_fnet/saved_models/' + Pretrained_model_name):
          shutil.rmtree('/content/gdrive/My Drive/pytorch_fnet/saved_models/' + Pretrained_model_name)
          shutil.copytree(Pretrained_model_folder,'/content/gdrive/My Drive/pytorch_fnet/saved_models/' + Pretrained_model_name)
        else:
          shutil.copytree(Pretrained_model_folder,'/content/gdrive/My Drive/pytorch_fnet/saved_models/' + Pretrained_model_name)


def split_validation_training(fnet_data, percent_for_validation, fnet_dataset_path, source_name, target_name, continue_training, new_datasource=True, replace_existing_dataset=True, Pretrained_model_name=None):
  #split validation data randomly for new training or use the same ones for previously trained datasets based on the csv file provided
  if continue_training==False or (continue_training==True and new_datasource==True):
      val_files = random.sample(source,round(len(source)*(percent_for_validation/100)))
      #Move a random set of files from the training to the validation folders
      for file in val_files:
        shutil.move(fnet_dataset_path+'/'+source_name+'/'+file, fnet_dataset_path+'/Validation_Input/'+file)
        shutil.move(fnet_dataset_path+'/'+target_name+'/'+file, fnet_dataset_path+'/Validation_Target/'+file)
  #use same validation datafiles as before
  elif continue_training==True and new_datasource==False:
      csv_file = "/content/gdrive/My Drive/pytorch_fnet/saved_models/" + Pretrained_model_name +"/"+Pretrained_model_name+"_val.csv"
      df = pd.read_csv(csv_file)
      list_df = df["path_signal"]
      for i in list_df:
        i_name = os.path.basename(i)
        shutil.move(fnet_dataset_path+'/'+source_name+'/'+i_name, fnet_dataset_path+'/Validation_Input/'+i_name)
        shutil.move(fnet_dataset_path+'/'+target_name+'/'+i_name, fnet_dataset_path+'/Validation_Target/'+i_name)
  return 
# val_files = split_validation_training(percent_for_validation,fnet_dataset_path, source_name, target_name)

def modify_train_sh():
    with open("/content/gdrive/My Drive/pytorch_fnet/scripts/train_model.sh", "r") as f:
      if not "gpu_ids ${GPU_IDS} \\" in f.read():
        replace("/content/gdrive/My Drive/pytorch_fnet/scripts/train_model.sh","       --gpu_ids ${GPU_IDS}","       --gpu_ids ${GPU_IDS} \\")

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


def create_csv_files(fnet_dataset_path, fnet_data_csv, dataset, val_signal, val_target, source_name, target_name, source, target, continue_training = False, replace_existing_dataset=False):
    if continue_training == False:
        if os.path.exists(fnet_data_csv +'/'+ dataset+'_val.csv'):
            os.remove(fnet_data_csv +'/'+dataset+'_val.csv')
        #Finally, we create a validation csv file to construct the validation dataset
        os.chdir(fnet_data_csv)
        with open(dataset+'_val.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["path_signal","path_target"])
            for i in range(0,len(val_signal)):
               writer.writerow([fnet_dataset_path+"/Validation_Input/"+val_signal[i], fnet_dataset_path +"/Validation_Target/"+val_target[i]])

        if os.path.exists(fnet_data_csv + '/'+dataset+'.csv'):
           os.remove(fnet_data_csv + '/'+dataset+'.csv')
        with open(dataset+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["path_signal","path_target"])
            for i in range(0,len(source)):
                writer.writerow([fnet_dataset_path+"/"+source_name+"/"+source[i], fnet_dataset_path+"/"+target_name+"/"+target[i]])
    elif continue_training == True and replace_existing_dataset == True:
          #Finally, we create a validation csv file to construct the validation dataset
          if os.path.exists(fnet_data_csv + '/'+dataset+'.csv'):
              os.remove(fnet_data_csv +'/'+dataset+'_val.csv')
              os.chdir(fnet_data_csv)
              with open(dataset+'_val.csv', 'w', newline='') as file:
                  writer = csv.writer(file)
                  writer.writerow(["path_signal","path_target"])
                  for i in range(0,len(val_signal)):
                    writer.writerow([fnet_dataset_path+"/Validation_Input/"+val_signal[i], fnet_dataset_path +"/Validation_Target/"+val_target[i]])

          if os.path.exists(fnet_data_csv + '/'+dataset+'.csv'):
              os.remove(fnet_data_csv + '/'+dataset+'.csv')
              os.chdir(fnet_data_csv)
              with open(dataset+'.csv', 'w', newline='') as file:
                  writer = csv.writer(file)
                  writer.writerow(["path_signal","path_target"])
                  for i in range(0,len(source)):
                      writer.writerow([fnet_dataset_path+"/"+source_name+"/"+source[i], fnet_dataset_path+"/"+target_name+"/"+target[i]])
      # add to the existing csv file the 
    elif continue_training == True and replace_existing_dataset == False:
            print("You use the same data as for the previous training")



def add_module_to_functions_py():  
    f = open("functions.py", "r")
    contents = f.readlines()
    f.close()
    f = open("functions.py", "r")
    if not 'import fnet.fnet_model' in f.read():
      contents.insert(5, 'import fnet.fnet_model')
    f.close()
    f = open("functions.py", "w")
    contents = "".join(contents)
    f.write(contents)
    f.close()

