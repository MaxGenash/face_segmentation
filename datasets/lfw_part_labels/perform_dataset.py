import os
from os import listdir, makedirs
from os.path import isfile, join, exists
import shutil
import glob

# Performs the dataset in convenient way

def getAllFileInFolder(folderPath, fileExtension):
    totalExtension = ''
    if fileExtension.startswith('.'):
        totalExtension = fileExtension
    else:
        totalExtension = '.' + fileExtension

    return [f for f in listdir(folderPath) if isfile(join(folderPath, f)) and f.endswith(totalExtension)]


def getAllFoldersInFolder(folderPath):
    return [f for f in listdir(folderPath) if not isfile(join(folderPath, f))]


def remove_pictures_without_masks(images_dir='./raw/images/', masks_dir='./raw/masks/'):
    masks_set = set()

    print('\nmasks:')
    for mask_file in glob.glob(masks_dir + '*'):
        # print(mask_file)
        # print(mask_file[len(masks_dir): -4])
        masks_set.add(mask_file[len(masks_dir): -4])
    print(len(masks_set))

    print('\nimages:')
    for img_file in glob.glob(images_dir + '*'):
        # print (img_file)
        # print(img_file[len(images_dir): -4])
        if not img_file[len(images_dir): -4] in masks_set:
            print(img_file)
            os.remove(img_file)
    print(len(glob.glob(images_dir + '*')))


# Start
if True: #__name__ == '__main__':
    dataset_folder = 'lfw_funneled'
    # get all folder names
    folders = getAllFoldersInFolder(dataset_folder)
    # get all file names
    image_path = []
    image_name = []
    for folder in folders:
        subfolder_path = join(dataset_folder, folder)
        files = getAllFileInFolder(subfolder_path, 'jpg')
        for f in files:
            image_path.append(subfolder_path)
            image_name.append(f)

    # Generate the new folder structure
    new_folder_path = 'raw'
    if exists(new_folder_path):
        shutil.rmtree(new_folder_path)

    dst_path = join(new_folder_path, 'images')
    makedirs(dst_path)
    # copy all the images
    for src_path, src_image in zip(image_path, image_name):
        src = join(src_path, src_image)
        dst = join(dst_path, src_image)
        shutil.copyfile(src, dst)

    remove_pictures_without_masks()

    print("Job finished")
