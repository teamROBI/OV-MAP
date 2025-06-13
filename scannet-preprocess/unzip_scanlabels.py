import os
import zipfile
import shutil

# Specify the folder path where the zip files are located
folder_path = '/media/tidy/Extreme SSD/ScanNet/dataset/scans'
directory_path = '/media/tidy/Extreme SSD/ScanNet/rgbd_val_dataset'


def mkdir(path):
    # Check if the directory already exists
    if not os.path.exists(path):
        # Create the directory
        os.makedirs(path)
    else:
        print(f"Directory '{path}' already exists.")
        return 'pass'

def unzip_to_dest(zip_path, dest_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_path)

# Function to recursively search for the zip file

def find_zip_file(directory_path, folder, val_scenes):
    for root, dirs, files in os.walk(folder):
        if os.path.basename(root) in val_scenes:
            for file in files:
                if file.endswith('.sens'):
                    scene_name = root.split('/')[-1]
                    dest_dir = os.path.join(directory_path, scene_name)
                    exist = mkdir(dest_dir) # make new directory
                    if exist == 'pass':
                        continue
                    sens_file = scene_name+'.sens'
                    sens_file_path = os.path.join(root, sens_file)
                    shutil.copy(sens_file_path, dest_dir+'/') # copy .sens file
                    zip_label_path = os.path.join(root, scene_name+'_2d-label.zip')
                    unzip_to_dest(zip_label_path, dest_dir)
                    zip_label_filt_path = os.path.join(root, scene_name+'_2d-label-filt.zip')
                    unzip_to_dest(zip_label_filt_path, dest_dir)

# If only validataion dataset
with open('meta_data/scannetv2_val_orig.txt') as val_file:
    val_scenes = val_file.read().splitlines()

# print(len(val_scenes))
find_zip_file(directory_path, folder_path, val_scenes)