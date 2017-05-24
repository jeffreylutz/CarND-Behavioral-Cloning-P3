import csv
import cv2
import numpy as np
import math


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def trans_image(image, steer, trans_range):
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    # tr_y = 0
    (cols, rows) = image.shape[:2]
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))
    return (image_tr, steer_ang)


def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


def flip_image(image):
    return np.fliplr(image)


new_size_col,new_size_row = 64, 64
new_size_col,new_size_row = 200, 66
new_size_col, new_size_row = 320, 160


def preprocessImage(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row),interpolation=cv2.INTER_AREA)
    #image = image/255.-.5
    return image


lines = []


def load_csv(csv_filename):
    if(len(lines) > 0):
        print('Already built....')
        return lines
    with open(csv_filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def gen_data(csv_filename, batch_size = 32):
    lines = load_csv(csv_filename)
    images = np.zeros((batch_size, new_size_row, new_size_col,3))
    angles = np.zeros(batch_size)
    while 1:
        for i in range(batch_size):
            line_num = np.random.randint(len(lines))
            left_right_center = np.random.randint(3)
            line = lines[line_num]
            if(left_right_center == 0):
                path_file = line[2].strip()
                shift_ang = .25
            if(left_right_center == 1):
                path_file = line[0].strip()
                shift_ang = 0.
            if(left_right_center == 2):
                path_file = line[1].strip()
                shift_ang = -.25
            angle = float(line[3]) + shift_ang
            image = cv2.imread(path_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image, angle = trans_image(image, angle, 100)
            image = augment_brightness_camera_images(image)
            image = preprocessImage(image)
            image = np.array(image)
            if np.random.randint(2) == 0:
                image = flip_image(image)
                angle = -1. * angle
                images[i] = image
                angles[i] = angle
        yield images, angles
