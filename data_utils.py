import csv
import cv2
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)
# Set new size
new_size_col, new_size_row = 64,64


def load_resized_data(csv_filename):
    X_data, y_data = load_data(csv_filename)
    for i in range(len(X_data)):
        X_data[i] =  cv2.resize(X_data[i], (new_size_col, new_size_row), interpolation=cv2.INTER_AREA)
    return X_data, y_data


def load_data(csv_filename):
    lines = []
    with open(csv_filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    angles = []
    for line in lines:
        ## Only use center images- index=0
        filename = line[0]

        angle = float(line[3])
        ## Only consider 25% of driving straight
        # if angle > -.05 and angle < .05 and np.random.randint(6):
        if angle > -.05 and angle < .05:
             continue

        image = cv2.imread(line[0])


        ## Convert image from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ## Crop image to exclude the sky and front of car
        # image = image[70:(160 - 25), 0:320]

        images.append(image)
        angles.append(angle)
        # # Resize to (32, 16, 3)
        # print(image.shape)
        # image = image.resize(16, 32, 3)
        # print(image.shape())
        # # Convert to grayscale
        # image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        # Add flipped image along with flipped angle
        image = np.fliplr(image)
        angle = -1.0 * angle
        images.append(image)
        angles.append(angle)
    return np.array(images), np.array(angles)


def resize(imgs, shape=(32, 16, 3)):
    """
    Resize images to shape.
    """
    height, width, channels = shape
    imgs_resized = np.empty([len(imgs), height, width, channels])
    for i, img in enumerate(imgs):
        imgs_resized[i] = cv2.imresize(img, shape)

    return imgs_resized


def rgb2gray(imgs):
    """
    Convert images to grayscale.
    """
    return np.mean(imgs, axis=3, keepdims=True)


def normalize(imgs):
    """
    Normalize images between [-1, 1].
    """
    return imgs / (255.0 / 2) - 1


def preprocess(imgs,angles):
    imgs_processed = resize(imgs)
    imgs_processed = rgb2gray(imgs_processed)
    imgs_processed = normalize(imgs_processed)
    return imgs_processed


def random_flip(imgs, angles):
    new_imgs = np.empty_like(imgs)
    new_angles = np.empty_like(angles)
    for i, (img, angle) in enumerate(zip(imgs, angles)):
        if np.random.choice(2):
            new_imgs[i] = np.fliplr(img)
            new_angles[i] = angle * -1
        else:
            new_imgs[i] = img
            new_angles[i] = angle
    return new_imgs, new_angles


def augment(imgs, angles):
    imgs_augmented, angles_augmented = random_flip(imgs, angles)
    return imgs_augmented, angles_augmented


def read_imgs(img_paths):
    imgs = np.empty([len(img_paths), 160, 320, 3])
    for i, path in enumerate(img_paths):
        imgs[i] = cv2.imread(path)
    return imgs


def gen_batches(imgs, angles, batch_size):
    num_elts = len(imgs)

    while True:
        indices = np.random.choice(num_elts, batch_size)
        batch_imgs_raw, angles_raw = read_imgs(imgs[indices]), angles[indices].astype(float)
        batch_imgs, batch_angles = preprocess(batch_imgs_raw, angles_raw)
        yield batch_imgs, batch_angles
