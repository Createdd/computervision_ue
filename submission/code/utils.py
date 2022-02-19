### imports

import skimage
from skimage import io
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
import os
import matplotlib.patches as patches
import matplotlib
import itertools


### functions related to image processing

def warp_image(img, mask, homography):
    
    '''
    Applies the provided mask to the image to remove the timestamp and warps the image to the perspective of the center camera 
    with the help of the provided homography matrix.
    ----------
    Parameters: 
        img: image in rgba format
        mask: binary image
        homography: 3x3 array
    ----------
    Returns:
        processed_img: masked and warped image in rgb format
    '''
    
    img = img[..., :3].astype(np.float32)/255
    img[mask==0] = np.nan
    warped_img = img.copy()
    for dim in range(3):
        warped_img[:,:,dim] = cv2.warpPerspective(img[:,:,dim],homography,img.shape[:2],borderMode = cv2.BORDER_CONSTANT,borderValue = np.nan) 
    
    return warped_img



def integrate_images(img_list):
    
    '''
    Averages all images in img_list and removes pixels from the average image that are not covered by all input images.
    ----------
    Parameters: 
        img_list: list of images in rgb format
    ----------
    Returns:
        integrated_img_nan: average image in rgb format
    '''
    
    integrated_img = np.nanmean(np.array(img_list), axis = 0)
    integrated_img_nan = integrated_img.copy()
    for dim in range(3):
        integrated_img_nan[:,:,dim] = np.where(np.isnan(img_list).any(axis = 0).all(axis = 2), np.nan, integrated_img[:,:,dim])
    
    return integrated_img_nan



def crop_image(img,coords = [370,140,315,700]):
    
    '''
    Crops images to the size provided in the coords argument.
    ----------
    Parameters: 
        img: image in rgb format
        coords: list with 4 elements determining y-coordinate, x-coordinate, height and width of the cropped area, default: [370,140,315,700]
    ----------
    Returns:
        cropped_img: cropped image in rgb format
    '''
    
    [y,x,h,w] = coords
    cropped_img = img[y:y+h,x:x+w,:]
    
    return cropped_img



def subtract_variance_channels(var_image):
    
    '''
    Subtracts the green channel from the blue and the red channel, respectively, of the input image, 
    sets all negative pixel values to 0 and scales the resulting grayscale image up by a factor of 500.
    ----------
    Parameters: 
        var_image: image with 3 channels and float values in an arbitrary range
    ----------
    Returns:
        var_blue_green: scaled grayscale image of blue minus green channel  
        var_red_green: scaled grayscale image of red minus green channel  
    '''
    
    var_red = var_image[...,0]
    var_green = var_image[...,1]
    var_blue  = var_image[...,2]

    var_blue_green = var_blue - var_green
    var_blue_green = np.where(var_blue_green<0, 0, var_blue_green)
    var_blue_green *= 500

    var_red_green = var_red - var_green
    var_red_green = np.where(var_red_green<0, 0, var_red_green)
    var_red_green *= 500

    return var_blue_green, var_red_green



def apply_brightness_contrast(img, brightness = 0, contrast = 0):
    
    '''
    Adjusts brightness and contrast of the input image.
    Adapted from: https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    ----------
    Parameters: 
        img: image with pixel values in the range of 0 to 1 (e.g. grayscale)
        brightness: float between -0.5 and 0.5, default: 0
        contrast: float between -0.5 and 0.5, default: 0
    ----------
    Returns:
        adjusted_img: image adjusted for brightness and contrast  
    '''   
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 1
        else:
            shadow = 0
            highlight = 1 + brightness
        alpha_b = highlight - shadow
        gamma_b = shadow
        
        adjusted_img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    else:
        adjusted_img = img.copy()
    
    if contrast != 0:
        alpha_c = 131*(contrast*255 + 127)/(127*(131-contrast*255))
        gamma_c = 0.5*(1-alpha_c)
        
        adjusted_img = cv2.addWeighted(adjusted_img, alpha_c, adjusted_img, 0, gamma_c)

    return adjusted_img



def create_sample_images(folder_path, mask):
    
    '''
    Takes the path to the folder of one sample consisting of 70 images as input, pre-processes the images,
    integrates them, calculates the variance between time points, subtracts the green channel of the variance image
    from the red and the blue channel, respectively, adjusts brightness and contrast of the resulting images, 
    dilates, erodes and dilates them again using the previously defined functions. It returns the resulting images
    after the second dilation as well as a dictionary containing all intermediate images adjusted for plotting in rgb format.
    ----------
    Parameters: 
        folder_path: str, path to the folder containing the 70 images of one sample
        mask: binary image
    ----------
    Returns:
        dilated2_blue: processed image resulting from the blue-green channel of the variance image
        dilated2_red: processed image resulting from the red-green channel of the variance image
        image dict: dictionary containing the warped raw image of the center camera at timepoint 3 ('warped_G01_3'),
            the integrated image at timepoint 3 uncropped ('intergrated_3') and cropped ('cropped_3'),
            the image of the variance between the timepoints ('var_image'), the subtracted variance channel images
            ('var_blue_green', 'var_red_green'), adjusted for brightness and contrast ('adjusted_blue', 'adjusted_red'),
            dilated ('dilated_blue', 'dilated_red'), eroded ('eroded_blue', 'eroded_red') and dilated again
            ('dilated2_blue', 'dilated2_red') using a 5x5- and a 7x7-kernel, respectively, for dilation and erosion 
            ready for plotting in rgb format
    ----------
    Required functions:
        preprocess_image(img, mask, homography)
        integrate_images(img_list)
        crop_image(img)
        subtract_variance_channels(var_image)
        apply_brightness_contrast(img, contrast, brightness)
    '''    
        
    image_dict = {}
    
    with open(os.path.join(folder_path,'homographies.json')) as f:
        homographies = json.load(f)
        
    cropped_images = []
    for timepoint in [0,1,2,3,4,5,6]:
        warped_images = []
        for camera in ['B01', 'B02', 'B03', 'B04', 'B05', 'G01', 'G02', 'G03', 'G04', 'G05']:
            image = skimage.io.imread(os.path.join(folder_path,str(timepoint)+'-'+camera+'.png'))
            homography = np.array(homographies[str(timepoint) + '-' + camera])
            warped_image = warp_image(image, mask, homography)
            if camera == 'G01' and timepoint == 3:
                warped_G01_3 = warped_image
            warped_images.append(warped_image)
        integrated_image = integrate_images(warped_images)
        cropped_image = crop_image(integrated_image)
        cropped_images.append(cropped_image)
        if timepoint == 3:
            intergrated_3 = integrated_image
            cropped_3 = cropped_image
    
    var_image = np.nanvar(cropped_images, axis = 0)
    var_blue_green, var_red_green = subtract_variance_channels(var_image)
    
    adjusted_blue = apply_brightness_contrast(var_blue_green, 0.25, 0.5)
    adjusted_red = apply_brightness_contrast(var_red_green, 0.25, 0.5)

    dilate_kernel = np.ones((5, 5), 'uint8')
    erode_kernel = np.ones((7, 7), 'uint8')

    dilated_blue = cv2.dilate(adjusted_blue, dilate_kernel, iterations = 1)
    eroded_blue = cv2.erode(dilated_blue, erode_kernel)
    dilated2_blue = cv2.dilate(eroded_blue, dilate_kernel, iterations = 1)
    
    dilated_red = cv2.dilate(adjusted_red, dilate_kernel, iterations = 1)
    eroded_red = cv2.erode(dilated_red, erode_kernel)
    dilated2_red = cv2.dilate(eroded_red, dilate_kernel, iterations = 1)
       
    image_dict['warped_G01_3'] = warped_G01_3
    image_dict['intergrated_3'] = intergrated_3
    image_dict['cropped_3'] = cropped_3
    image_dict['var_image'] = var_image/np.max(var_image)
    image_dict['var_red'] = np.moveaxis(np.array(3*[var_image[...,0]/np.max(var_image[...,0])]),0,2).copy()
    image_dict['var_green'] = np.moveaxis(np.array(3*[var_image[...,1]/np.max(var_image[...,1])]),0,2).copy()
    image_dict['var_blue'] = np.moveaxis(np.array(3*[var_image[...,2]/np.max(var_image[...,2])]),0,2).copy()
    image_dict['var_blue_green'] = np.moveaxis(np.array(3*[var_blue_green/np.max(var_blue_green)]),0,2).copy()
    image_dict['var_red_green'] = np.moveaxis(np.array(3*[var_red_green/np.max(var_red_green)]),0,2).copy()
    image_dict['adjusted_blue'] = np.moveaxis(np.array(3*[np.where(adjusted_blue<0,0,np.where(adjusted_blue>1,1,adjusted_blue))]),0,2).copy()
    image_dict['adjusted_red'] = np.moveaxis(np.array(3*[np.where(adjusted_red<0,0,np.where(adjusted_red>1,1,adjusted_red))]),0,2).copy()
    image_dict['dilated_blue'] = np.moveaxis(np.array(3*[np.where(dilated_blue<0,0,np.where(dilated_blue>1,1,dilated_blue))]),0,2).copy()
    image_dict['dilated_red'] = np.moveaxis(np.array(3*[np.where(dilated_red<0,0,np.where(dilated_red>1,1,dilated_red))]),0,2).copy()
    image_dict['eroded_blue'] = np.moveaxis(np.array(3*[np.where(eroded_blue<0,0,np.where(eroded_blue>1,1,eroded_blue))]),0,2).copy()
    image_dict['eroded_red'] = np.moveaxis(np.array(3*[np.where(eroded_red<0,0,np.where(eroded_red>1,1,eroded_red))]),0,2).copy()
    image_dict['dilated2_blue'] = np.moveaxis(np.array(3*[np.where(dilated2_blue<0,0,np.where(dilated2_blue>1,1,dilated2_blue))]),0,2).copy()
    image_dict['dilated2_red'] = np.moveaxis(np.array(3*[np.where(dilated2_red<0,0,np.where(dilated2_red>1,1,dilated2_red))]),0,2).copy()

    return dilated2_blue, dilated2_red, image_dict


### functions related to getting boundary boxes

def get_bounding_boxes(binary_img):
    
    '''
    Creates bounding boxes for all spots in the binary input image.
    ----------
    Parameters: 
        binary_img: binary image with pixel values of 0 or 1
    ----------
    Returns:
        bounding_boxes: list of coordinate lists for each bounding box containing x-coordinate, y-coordinate, width and height
    '''
    
    contours, hierarchy = cv2.findContours(binary_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        bounding_boxes.append([x,y,w,h])
        
    return bounding_boxes



def merge_bounding_boxes(bounding_boxes):
    
    '''
    Increases the size of each bounding box by 2 pixels in each direction, merges overlapping bounding boxes, 
    increases the resulting bounding boxes by 14 pixels in width and 8 pixels in height and sorts them by area.
    ----------
    Parameters: 
        bounding_boxes: list of coordinate lists for each bounding box containing x-coordinate, y-coordinate, width and height
    ----------
    Returns:
        merged_bounding_boxes: sorted list of coordinate lists for each merged bounding box containing x-coordinate, y-coordinate, width and height
    '''    
    
    bounding_boxes = [[sum(x) for x in zip(bounding_boxes[i], [-2,-2,4,4])] for i in range(len(bounding_boxes))]
    
    for k in range(len(bounding_boxes)):
        box1 = bounding_boxes[k]
        pixel_list1 = np.array([[i,j] for i,j in list(itertools.product(list(range(box1[0], box1[0]+box1[2]+1)), list(range(box1[1], box1[1]+box1[3]+1))))])
        max1 = np.max(np.array(pixel_list1), axis=0)
        for l in range(len(bounding_boxes)):
            if k!=l:
                box2 = bounding_boxes[l]
                pixel_list2 = np.array([[i,j] for i,j in list(itertools.product(list(range(box2[0], box2[0]+box2[2]+1)), list(range(box2[1], box2[1]+box2[3]+1))))])
                max2 = np.max(np.array(pixel_list2), axis=0)
                common_x = list(set(pixel_list1[:,0]).intersection(set(pixel_list2[:,0])))
                common_y = list(set(pixel_list1[:,1]).intersection(set(pixel_list2[:,1])))
                if common_x != [] and common_y != []:
                    merge_box = [min(box1[0], box2[0]), min(box1[1], box2[1]), max(max1[0], max2[0])-min(box1[0], box2[0]), max(max1[1], max2[1])-min(box1[1], box2[1])]
                    bounding_boxes[k] = merge_box
                    bounding_boxes[l] = merge_box

    merged_bounding_boxes = []
    for box in bounding_boxes:
        if box not in merged_bounding_boxes:
            merged_bounding_boxes.append([sum(x) for x in zip(box, [-7,-4,14,8])])
            
    merged_bounding_boxes.sort(key=lambda x: x[2] * x[3], reverse=True)
    
    return merged_bounding_boxes



def select_bounding_boxes(merged_bounding_boxes_blue, merged_bounding_boxes_red):
    
    '''
    Returns the biggest bounding box for the blue-minus-green and the red-minus-green variance image, respectively,
    if they are bigger than 24 pixels in width and 16 pixels in height. If the resulting bounding boxes are overlapping, 
    only the bigger one is returned. Bounding boxes smaller then 38 pixels in width and 30 pixels in height are padded
    to this size.
    ----------
    Parameters: 
        merged_bounding_boxes_blue: sorted list of coordinate lists for each merged bounding box containing x-coordinate, y-coordinate, width and height of the blue-minus-green variance image
        merged_bounding_boxes_red: same as above for the red-minus-green variance image
    ----------
    Returns:
        detections: list of up to 2 coordinate lists of bounding boxes
    '''    
     
    detections = []

    if merged_bounding_boxes_blue != []:
        detection_blue = merged_bounding_boxes_blue[0]
        if detection_blue[2]>24 and detection_blue[3]>16:
            detections.append([int(detection_blue[0]+140),int(detection_blue[1]+370),int(detection_blue[2]),int(detection_blue[3])])

    if merged_bounding_boxes_red != []:
        detection_red = merged_bounding_boxes_red[0]
        if detection_red[2]>24 and detection_red[3]>16:
            detections.append([int(detection_red[0]+140),int(detection_red[1]+370),int(detection_red[2]),int(detection_red[3])])

    if len(detections) == 2:
        
        d1 = detections[0]
        d2 = detections[1]
        x = max(d1[0], d2[0])
        y = max(d1[1], d2[1])
        xx = min(d1[0] + d1[2], d2[0] + d2[2])
        yy = min(d1[1] + d1[3], d2[1] + d2[3])
        intersection_area = max(0, xx-x) * max(0, yy-y)
        
        if intersection_area > 0:
            detections.sort(key=lambda x: x[2]*x[3], reverse=True)
            detections = [detections[0]]
        else:
            detections.sort(key=lambda x: x[1])
            
    for detection in detections:
        if detection[2] < 38:
            x_pad = int(35-detection[2])/2
            detection[0] = int(detection[0] - x_pad)
            detection[2] = int(detection[2] + 2*x_pad)
        if detection[3] < 30:
            y_pad = int(30-detection[3])/2
            detection[1] = int(detection[1] - y_pad)
            detection[3] = int(detection[3] + 2*y_pad)

    return detections



def get_detections(dilated2_blue, dilated2_red, binary_threshold = 1):
    
    '''
    Creates a binary image from the two input images using the giveen binary threshold and returns detections
    determined by the functions get_bounding_boxes(), merge_bounding_boxes() and select_bounding_boxes().
    ----------
    Parameters: 
        dilated2_blue: processed image resulting from the blue-green channel of the variance image
        dilated2_red: processed image resulting from the red-green channel of the variance image
        binary_threshold: int or float, threshold for creating the binary image, default: 1
    ----------
    Returns:
        detections: list of up to 2 coordinate lists of bounding boxes
    ----------
    Required functions:
        get_bounding_boxes(binary_img)
        merge_bounding_boxes(bounding_boxes)
        select_bounding_boxes(merged_bounding_boxes_blue, merged_bounding_boxes_red)
    '''

    binary_blue = np.where(dilated2_blue>=binary_threshold, 1, 0).astype('uint8')
    binary_red = np.where(dilated2_red>=binary_threshold, 1, 0).astype('uint8')

    bounding_boxes_blue = get_bounding_boxes(binary_blue)
    bounding_boxes_red = get_bounding_boxes(binary_red)

    merged_bounding_boxes_blue = merge_bounding_boxes(bounding_boxes_blue)
    merged_bounding_boxes_red = merge_bounding_boxes(bounding_boxes_red)

    detections = select_bounding_boxes(merged_bounding_boxes_blue, merged_bounding_boxes_red)
    
    return detections, binary_blue, binary_red



def iterative_detection(dilated2_blue, dilated2_red):
    
    '''
    Gets detections for the two processed input images using the previousyl defined function get_detections()
    and iteratively decreases the binary threshold when no detections can be retrived.
    ----------
    Parameters: 
        dilated2_blue: processed image resulting from the blue-green channel of the variance image
        dilated2_red: processed image resulting from the red-green channel of the variance image
    ----------
    Returns:
        detections: list of up to 2 coordinate lists of bounding boxes
    ----------
    Required functions:
        get_detections(dilated2_blue, dilated2_red, binary_threshold)
    '''
    
    binary_threshold = 1
    detections, binary_blue, binary_red = get_detections(dilated2_blue, dilated2_red, binary_threshold = binary_threshold)
    
    while detections == []:
        binary_threshold -= 1
        detections, binary_blue, binary_red = get_detections(dilated2_blue, dilated2_red, binary_threshold = binary_threshold)

    binary_blue_plot = np.moveaxis(np.array(3*[binary_blue]),0,2).astype(float).copy()
    binary_red_plot = np.moveaxis(np.array(3*[binary_red]),0,2).astype(float).copy()
        
    return detections, binary_blue_plot, binary_red_plot



def create_detection_dict(dataset, data_dir):
    
    '''
    Determines detections for all samples in the specified data set, returns them as a dictionary and saves it in a json file.
    ----------
    Parameters: 
        dataset: str, e.g. 'train', 'test' or 'validation'
        data_dir: relative or absolute path to the directory in which the raw data is stored, must contain
            a folder with the name of the dataset, containing subfolders starting with the name of the dataset
            (or with 'valid' in case of the validation dataset), each containing 70 images with names ranging from 
            '0-B01' to '6-G05'
    ----------
    Returns:
        detection_dict: dictionary with folder names (e.g. 'valid-2-0') as keys and lists of bounding box coordinates ([x,y,w,h]) as values
    ----------
    Required functions:  
        create_sample_images(folder_path, mask) 
        iterative_detection(dilated2_blue, dilated2_red)
    '''
    
    mask = skimage.io.imread(os.path.join(data_dir,'mask.png'))
    detection_dict = {}
    
    for folder in os.listdir(os.path.join(data_dir, dataset)): 
        if folder.startswith(dataset) or (dataset == 'validation' and folder.startswith('valid')):
            
            folder_path = os.path.join(data_dir, dataset, folder)
            dilated2_blue, dilated2_red, image_dict = create_sample_images(folder_path, mask)        
            detections, binary_blue_plot, binary_red_plot = iterative_detection(dilated2_blue, dilated2_red)

            detection_dict[folder] = detections

    if dataset == 'validation':
        detection_file_name = 'val.json'
    else:
        detection_file_name = dataset + '.json'

    with open(os.path.join('..',detection_file_name), 'w') as detection_file:
        json.dump(detection_dict, detection_file)
    
    print('Detection dictionary was successfully saved.')
    
    return detection_dict

    
### evaluation utilities provided in the course repository

from typing import Dict, NewType, List, Tuple, Union
import pathlib

Path = Union[pathlib.Path, str]
BoundingBox = NewType("BoundingBox", Tuple[int, int, int, int])
YoloBox = NewType("YoloBox", Tuple[float, float, float, float])
Shape = NewType("Shape", Tuple[int, int])



def compute_AP(detections: Dict[str, List[BoundingBox]],
               targets: Dict[str, List[BoundingBox]]) -> float:
    """ Compute the average precision.

    Params:
        detections: list of detected bounding boxes within each sample
        targets: list of ground truth bounding boxes within each sample
    """
    # define the IoU threshold sequence
    thresholds = np.arange(0.1, 1.0, 0.1)

    precision = np.zeros_like(thresholds)
    recall = np.zeros_like(thresholds)

    iou_scores = [compute_IoU(detections[k], targets[k])
                  for k in targets.keys()]

    for i, iou_th in enumerate(thresholds):
        true_positives = sum(
            [np.sum(np.any(iou > iou_th, 1)) for iou in iou_scores])
        false_positives = sum(
            [np.sum(~np.any(iou > iou_th, 1)) for iou in iou_scores])
        false_negatives = sum(
            [np.sum(~np.any(iou > iou_th, 0)) for iou in iou_scores])

        if true_positives + false_positives:
            precision[i] = true_positives/(true_positives+false_positives)
        else:
            precision[i] = 0
        recall[i] = true_positives/(true_positives+false_negatives)

    # compute average precision
    recall = np.append(recall, 0)
    ap = np.sum((recall[:-1] - recall[1:]) * precision)
    return ap



def compute_IoU(detections: List[BoundingBox],
                targets: List[BoundingBox]) -> np.array:
    """ Compute the intersection of union (IoU) score.

    Params:
        detections: detected bounding boxes
        targets: ground truth bounding boxes

    Return:
        Array of IoU score between each pair of detected and target bounding
        box, where the detections are along the rows and the targets along
        the columns.
    """
    iou = np.empty((len(detections), len(targets)))

    for i, d in enumerate(detections):
        dx, dy, dw, dh = d
        for j, t in enumerate(targets):
            tx, ty, tw, th = t
            x = max(dx, tx)
            y = max(dy, ty)
            xx = min(dx + dw, tx + tw)
            yy = min(dy + dh, ty + th)
            intersection_area = max(0, xx-x) * max(0, yy-y)
            iou[i, j] = intersection_area / (dw*dh + tw*th - intersection_area)
    return iou


def read_bb(file: Path) -> Dict[str, List[BoundingBox]]:
    """ Read bounding boxes from json file.
    """
    with open(file) as f:
        js = json.load(f)
    return js


def write_bb(file: Path, bbs: Dict[str, List[BoundingBox]]) -> None:
    """ Write bounding boxes to json file.
    """
    with open(file, "w") as f:
        json.dump(bbs, f)


### functions related to plotting images

def label_image(img, labels, shift = False, colour = (1,1,1), thickness = 2):
    
    '''
    Draws rectangle based on provided labels on the input image.
    ----------
    Parameters: 
        img: image in any format
        labels: list of bounding box coordinate lists containing x-coordinate, y-coordinate, width and height
        shift: bool, set True for cropped images and False for images in original size, default: False
        colour: tuple determining the colour of the drawn bounding boxes, default: white (1,1,1)
        thickness: pixel width of drawn bounding boxes, default: 2
    ----------
    Returns:
        img: input image with added bounding boxes
    '''
    
    if shift:
        x_shift, y_shift = 140, 370
    else:
        x_shift, y_shift = 0, 0
    
    for label in labels:
        cv2.rectangle(img,(label[0]-x_shift,label[1]-y_shift),(label[0]+label[2]-x_shift,label[1]+label[3]-y_shift),colour,thickness)

    return img



def plot_images(image_list, dataset, data_dir, folders, label_targets = False, target_colour = (0,1,0), label_detections = True, detection_colour = (1,0,0), detection_path = None, plot = True, save = False):
    
    '''
    Plots and/or saves intermediate images from different processing steps optionally with target and/or detection labels. If label_targets and label_detections
    are both set to True, IoU values and the average precission (AP) of the dataset are printed.
    ----------
    Parameters: 
        image_list: list of image names of different processing steps with elements in 
            ['warped_G01_3','intergrated_3','cropped_3','var_image',
            'var_blue_green','var_red_green','adjusted_blue','adjusted_red','dilated_blue','dilated_red',
            'eroded_blue','eroded_red','dilated2_blue','dilated2_red','binary_blue','binary_red']
        dataset: str, e.g. 'train', 'test' or 'validation'
        data_dir: str, relative or absolute path to the directory in which the raw data is stored, must contain
            a folder with the name of the dataset, containing subfolders starting with the name of the dataset
            (or with 'valid' in case of the validation dataset), each containing 70 images with names ranging from 
            '0-B01' to '6-G05'
        folders: list of all folder names for samples to be plotted (os.listdir(os.path.join(data_dir, dataset)) to analyse all samples of the dataset) 
        label_targets: bool, specifies whether targets will be labelled in the plotted images, default: False
        target_colour: tuple of length 3 with entries between 0 and 1, specifies the colour of the target labels, default: green (0,1,0)
        label_detections: bool, specifies whether detections will be labelled in the plotted images, default: True
        detection_colour: tuple of length 3 with entries between 0 and 1, specifies the colour of the target labels, default: green (1,0,0)  
        detection_path: str, relative or absolute path to the json file containing the detection dictionary, default: None
        plot: bool, speficies whether the images will be plotted in the jupyter notebook, default: True
        save: bool, specifies whether the images will be saved in a folder named processed_images, default: False
    ----------
    Required functions:
        create_sample_images(folder_path, mask)
        iterative_detection(dilated2_blue, dilated2_red)
        label_image(img, labels, shift, colour, thickness)
    '''
    mask = skimage.io.imread(os.path.join(data_dir,'mask.png'))

    if label_targets:
        target_flag = '_targets'
        try:
            with open(os.path.join(data_dir, dataset, 'labels.json'), 'r') as target_file:
                target_dict = json.load(target_file)
        except:
            raise NameError('No target labels available for dataset ' + dataset + '.')
    else:
        target_flag = ''

    if label_detections:
        detection_flag = '_detections'
        with open(detection_path, 'r') as detection_file:
            detection_dict = json.load(detection_file)
    else:
        detection_flag = '_detections'

    for folder in folders: 
        if folder.startswith(dataset) or (dataset == 'validation' and folder.startswith('valid')):
            folder_path = os.path.join(data_dir, dataset, folder)

            dilated2_blue, dilated2_red, image_dict = create_sample_images(folder_path, mask)        
            detections, binary_blue_plot, binary_red_plot = iterative_detection(dilated2_blue, dilated2_red)
            image_dict['binary_blue'] = binary_blue_plot
            image_dict['binary_red'] = binary_red_plot

            shift_dict = {}
            for image in list(image_dict.keys())[:2]:
                shift_dict[image] = False
            for image in list(image_dict.keys())[2:]:
                shift_dict[image] = True       

            for image_name in image_list:
                image = image_dict[image_name]
                if label_targets:
                    label_image(image, target_dict[folder], shift = shift_dict[image_name], colour = target_colour, thickness = 2)
                if label_detections:
                    label_image(image, detection_dict[folder], shift = shift_dict[image_name], colour = detection_colour, thickness = 2)
                if plot:
                    plt.figure()
                    plt.imshow(image)
                    plt.title(folder+'_'+image_name)
                if save:
                    save_folder = os.path.join(data_dir,'..','processed_images',dataset,image_name)
                    os.makedirs(save_folder, exist_ok = True)
                    plt.imsave(os.path.join(save_folder,folder+'_'+image_name+target_flag+detection_flag+'.png'), image)

            if label_targets and label_detections:
                print('IoU for '+folder+': '+str(compute_IoU(detection_dict[folder], target_dict[folder])))
    
    if label_targets and label_detections:
        print('AP for '+dataset+': '+str(compute_AP(detection_dict, target_dict)))
