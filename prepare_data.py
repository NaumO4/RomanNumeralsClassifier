import os
import random
from shutil import copyfile

import numpy as np
from PIL import Image

N_LABELS = 8
DIR = 'dataset'
DIR_REAL_DATA = os.path.join(DIR, "real_data")
DIR_AUGMENT_DATA = os.path.join(DIR, "augment_data")
DIR_TRAIN_DATA = os.path.join(DIR, "train")
DIR_TEST_DATA = os.path.join(DIR, "test")
LABELS = range(1, 1 + N_LABELS)
REAL_IMAGES_SHAPE = (52, 52, 3)
IMAGE_SHAPE = (48, 48, 1)
BATCH_SIZE = 64


def create_empty_imgs(count=1):
    img = np.zeros(REAL_IMAGES_SHAPE, dtype=np.uint8) + 255
    im = Image.fromarray(img)
    for number in LABELS:
        path = os.path.join(DIR_REAL_DATA, str(number))
        if not os.path.exists(path):
            os.mkdir(path)
        start = len(os.listdir(path))
        for i in range(start, start + count):
            im.save(os.path.join(path, str(number) + "_" + str(i) + ".jpeg"))


def load_imgs_by_number(dir, number, shape=IMAGE_SHAPE):
    path = os.path.join(dir, str(number))
    imgs_name = os.listdir(path)
    imgs_arr = np.zeros((len(imgs_name),) + shape)
    for i, img in enumerate(imgs_name):
        img = Image.open(os.path.join(path, img))
        img_arr = np.array(img.getdata()).reshape(img.size[0], img.size[1], -1)
        imgs_arr[i] = img_arr
    return imgs_arr


def load_dataset(dir=DIR_REAL_DATA, shape=IMAGE_SHAPE):
    features = np.empty((0,) + shape)
    labels = np.empty((0,), dtype=np.int)
    for number in LABELS:
        images = load_imgs_by_number(dir, number)
        l = np.zeros(images.shape[0], dtype=np.int)
        l[:] = number
        features = np.concatenate((features, images), axis=0)
        labels = np.concatenate((labels, l), axis=0)
    return features / 255., to_onehot(labels - 1)


'''
load augmented data and split to train and test dataset
'''


def split_dataset():
    if not os.path.exists(DIR_TRAIN_DATA):
        os.mkdir(DIR_TRAIN_DATA)
    if not os.path.exists(DIR_TEST_DATA):
        os.mkdir(DIR_TEST_DATA)
    for number in LABELS:
        path = os.path.join(DIR_AUGMENT_DATA, str(number))
        path_train = os.path.join(DIR_TRAIN_DATA, str(number))
        path_test = os.path.join(DIR_TEST_DATA, str(number))
        if not os.path.exists(path_train):
            os.mkdir(path_train)
        if not os.path.exists(path_test):
            os.mkdir(path_test)
        imgs = os.listdir(path)
        random.shuffle(imgs)
        train_len = len(imgs) * 0.8
        n = 0
        for img in imgs:
            if n < train_len:
                copyfile(os.path.join(path, img), os.path.join(path_train, img))
            else:
                copyfile(os.path.join(path, img), os.path.join(path_test, img))
            n = n + 1


'''
make several different images from one
'''


def augment_img(img, count, flip=False):
    import PIL.ImageOps
    img = img.convert('L')
    img = PIL.ImageOps.invert(img)
    real_img = img
    imgs = list()
    for i in range(count):
        img = real_img
        # mirror
        if flip:
            if random.random() >= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() >= 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # rotation
        if count < 3:
            degree = random.randint(-30, 30)
        else:
            degree = 60 / count * i - 30
        img = img.rotate(degree)
        # crop
        crop = 4
        left = random.randint(0, crop)
        upper = random.randint(0, crop)
        img = img.crop((left, upper, img.size[1] - crop + left, img.size[0] - crop + upper))
        # img = img.resize((48, 48))
        img = img.point(lambda x: 0 if x < 128 else 255, '1')
        imgs.append(img)
    return imgs


'''
create dataset with augmented images
'''


def augment_imgs():
    # make dirs
    if not os.path.exists(DIR_AUGMENT_DATA):
        os.mkdir(DIR_AUGMENT_DATA)
    for number in LABELS:
        path = os.path.join(DIR_AUGMENT_DATA, str(number))
        if not os.path.exists(path):
            os.mkdir(path)
    for number in LABELS:
        path = os.path.join(DIR_AUGMENT_DATA, str(number))
        n = 0
        print(number)
        flip = 0 < number <= 3
        path_real_imgs = os.path.join(DIR_REAL_DATA, str(number))
        img_names = os.listdir(path_real_imgs)
        aug_imgs = list()
        for img_name in img_names:
            img = Image.open(os.path.join(path_real_imgs, img_name))
            aug_imgs.extend(augment_img(img, 10, flip))

        for aug_img in aug_imgs:
            aug_img.save(os.path.join(path, str(number) + "_" + str(n) + ".jpeg"))
            n += 1

        if number == 4 or number == 6:
            number2 = 4 if number == 6 else 6
            path2 = os.path.join(DIR_AUGMENT_DATA, str(number2))
            for aug_img in aug_imgs:
                aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
                aug_img.save(os.path.join(path2, str(number2) + "_" + str(n) + ".jpeg"))
                n += 1


def split_to_train_dev(X, Y, ratio=0.75):
    X, Y = shufle_data(X, Y)
    train_len = int(len(X) * ratio)
    X_train = X[:train_len]
    Y_train = Y[:train_len]
    X_dev = X[train_len:]
    Y_dev = Y[train_len:]
    return X_train, Y_train, X_dev, Y_dev


def load_test_dataset():
    return load_dataset(DIR_TEST_DATA, IMAGE_SHAPE)


def load_train_datatset():
    return load_dataset(DIR_TRAIN_DATA, IMAGE_SHAPE)


def to_onehot(Y):
    oh = np.zeros((Y.shape[0], N_LABELS))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh


def shufle_data(X, Y):
    permutation = np.random.permutation(len(X))
    return X[permutation], Y[permutation]


def batch(X, Y, batch_size=64):
    X, Y = shufle_data(X, Y)
    batches = list()
    for i in range(0, len(X), batch_size):
        end_of_batch = max(len(X), i + batch_size)
        batches.append((X[i:end_of_batch], Y[i:end_of_batch]))
    return batches


# load_real_imgs_by_number(DIR_REAL_DATA, 1)
if __name__ == '__main__':
    # create_empty_imgs(20)
    # augment_imgs()
    # split_dataset()
    os.mkdir(os.path.join("", "dataset"))
    print('ffd')
