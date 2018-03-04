from glob import glob
import os.path
import cv2

def purge_augmentation(data_folder):
    for f in glob(os.path.join(data_folder, 'image_2', 'equ_*.png')):
        os.remove(f)
    for f in glob(os.path.join(data_folder, 'image_2', 'flipped_*.png')):
        os.remove(f)
    for f in glob(os.path.join(data_folder, 'gt_image_2', 'flipped_*.png')):
        os.remove(f)


def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def add_images_of_histogram_equalization(data_folder, image_paths):
    for image_path in image_paths:
        img = cv2.imread(image_path)
        equ_img = histogram_equalization(img)
        new_img_name = 'equ_' + os.path.basename(image_path)
        cv2.imwrite(os.path.join(data_folder, 'image_2', new_img_name), equ_img)

def add_images_of_flip(data_folder, image_paths):
    for image_path in image_paths:
        img = cv2.imread(image_path)
        flipped_img = cv2.flip(img, 1)
        new_img_name = 'flipped_' + os.path.basename(image_path)
        cv2.imwrite(os.path.join(data_folder, new_img_name), flipped_img)


def augment_images():
    data_folder = 'data/data_road/training'
    purge_augmentation(data_folder)
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    gt_image_paths = glob(os.path.join(data_folder, 'gt_image_2', '*.png'))
    add_images_of_histogram_equalization(data_folder, image_paths)
    add_images_of_flip(os.path.join(data_folder, 'image_2'), image_paths)
    add_images_of_flip(os.path.join(data_folder, 'gt_image_2'), gt_image_paths)


augment_images()