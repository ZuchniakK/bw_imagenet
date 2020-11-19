import random
import math
from os.path import join

from PIL import Image
import numpy as np
import argparse
import random
import numpy as np
import scipy
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps

import random
# from keras.preprocessing.image import ImageDataGenerator


from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]

operations = {
    'ShearX': lambda img, magnitude: shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: translate_y(img, magnitude),
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: invert(img, magnitude),
    'Equalize': lambda img, magnitude: equalize(img, magnitude),
    'Solarize': lambda img, magnitude: solarize(img, magnitude),
    'Posterize': lambda img, magnitude: posterize(img, magnitude),
    'Contrast': lambda img, magnitude: contrast(img, magnitude),
    'Color': lambda img, magnitude: color(img, magnitude),
    'Brightness': lambda img, magnitude: brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: cutout(img, magnitude),
}

CIFAR_10_POLICIES = [
    ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
    ['Rotate', 0.7, 2, 'TranslateX', 0.3, 9],
    ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
    ['ShearY', 0.5, 8, 'TranslateY', 0.7, 9],
    ['AutoContrast', 0.5, 8, 'Equalize', 0.9, 2],
    ['ShearY', 0.2, 7, 'Posterize', 0.3, 7],
    ['Color', 0.4, 3, 'Brightness', 0.6, 7],
    ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
    ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
    ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
    ['Color', 0.7, 7, 'TranslateX', 0.5, 8],
    ['Equalize', 0.3, 7, 'AutoContrast', 0.4, 8],
    ['TranslateY', 0.4, 3, 'Sharpness', 0.2, 6],
    ['Brightness', 0.9, 6, 'Color', 0.2, 8],
    ['Solarize', 0.5, 2, 'Invert', 0, 0.3],
    ['Equalize', 0.2, 0, 'AutoContrast', 0.6, 0],
    ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
    ['Color', 0.9, 9, 'Equalize', 0.6, 6],
    ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
    ['Brightness', 0.1, 3, 'Color', 0.7, 0],
    ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
    ['TranslateY', 0.9, 9, 'TranslateY', 0.7, 9],
    ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
    ['Equalize', 0.8, 8, 'Invert', 0.1, 3],
    ['TranslateY', 0.7, 9, 'AutoContrast', 0.9, 1],
]

IMAGENET_POLICIES = [
    ['Posterize', 0.4, 8, 'Rotate', 0.6, 9],
    ['Solarize', 0.6, 5, 'AutoContrast', 0.6, 5],
    ['Equalize', 0.8, 8, 'Equalize', 0.6, 3],
    ['Posterize', 0.6, 7, 'Posterize', 0.6, 6],
    ['Equalize', 0.4, 7, 'Solarize', 0.2, 4],
    ['Equalize', 0.4, 4, 'Rotate', 0.8, 8],
    ['Solarize', 0.6, 3, 'Equalize', 0.6, 7],
    ['Posterize', 0.8, 5, 'Equalize', 1.0, 2],
    ['Rotate', 0.2, 3, 'Solarize', 0.6, 8],
    ['Equalize', 0.6, 8, 'Posterize', 0.4, 6],
    ['Rotate', 0.8, 8, 'Color', 0.4, 0],
    ['Rotate', 0.4, 9, 'Equalize', 0.6, 2],
    ['Equalize', 0.0, 7, 'Equalize', 0.8, 8],
    ['Invert', 0.6, 4, 'Equalize', 1.0, 8],
    ['Color', 0.6, 4, 'Contrast', 1.0, 8],
    ['Rotate', 0.8, 8, 'Color', 1.0, 2],
    ['Color', 0.8, 8, 'Solarize', 0.8, 7],
    ['Sharpness', 0.4, 7, 'Invert', 0.6, 8],
    ['ShearX', 0.6, 5, 'Equalize', 1.0, 9],
    ['Color', 0.4, 0, 'Equalize', 0.6, 3],
    ['Equalize', 0.4, 7, 'Solarize', 0.2, 4],
    ['Solarize', 0.6, 5, 'AutoContrast', 0.6, 5],
    ['Invert', 0.6, 4, 'Equalize', 1.0, 8],
    ['Color', 0.6, 4, 'Contrast', 1.0, 8],
    ['Equalize', 0.8, 8, 'Equalize', 0.6, 3],
]


def apply_policy(img, policy):
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
    if random.random() < policy[4]:
        img = operations[policy[3]](img, policy[5])

    return img


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix


def shear_x(img, magnitude):
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    return img


def shear_y(img, magnitude):
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]), 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    return img


def translate_x(img, magnitude):
    magnitudes = np.linspace(-150 / 331, 150 / 331, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [0, 1,
                                  img.shape[1] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    return img


def translate_y(img, magnitude):
    magnitudes = np.linspace(-150 / 331, 150 / 331, 11)

    transform_matrix = np.array(
        [[1, 0, img.shape[0] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])],
         [0, 1, 0],
         [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    return img


def rotate(img, magnitude):
    magnitudes = np.linspace(-30, 30, 11)

    theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    return img


def auto_contrast(img, magnitude):
    img = Image.fromarray(img)
    img = ImageOps.autocontrast(img)
    img = np.array(img)
    return img


def invert(img, magnitude):
    img = Image.fromarray(img)
    img = ImageOps.invert(img)
    img = np.array(img)
    return img


def equalize(img, magnitude):
    img = Image.fromarray(img)
    img = ImageOps.equalize(img)
    img = np.array(img)
    return img


def solarize(img, magnitude):
    magnitudes = np.linspace(0, 256, 11)

    img = Image.fromarray(img)
    img = ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    img = np.array(img)
    return img


def posterize(img, magnitude):
    magnitudes = np.linspace(4, 8, 11)

    img = Image.fromarray(img)
    img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))))
    img = np.array(img)
    return img


def contrast(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)

    img = Image.fromarray(img)
    img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    img = np.array(img)
    return img


def color(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)

    img = Image.fromarray(img)
    img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    img = np.array(img)
    return img


def brightness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)

    img = Image.fromarray(img)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    img = np.array(img)
    return img


def sharpness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)

    img = Image.fromarray(img)
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    img = np.array(img)
    return img


def cutout(org_img, magnitude=None):
    magnitudes = np.linspace(0, 60 / 331, 11)

    img = np.copy(org_img)
    mask_val = img.mean()

    if magnitude is None:
        mask_size = 16
    else:
        mask_size = int(round(img.shape[0] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])))
    top = np.random.randint(0 - mask_size // 2, img.shape[0] - mask_size)
    left = np.random.randint(0 - mask_size // 2, img.shape[1] - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    img[top:bottom, left:right, :].fill(mask_val)

    return img


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BWImageNetDataGenerator:
    def __init__(self, directory, batch_size=32, target_size=(224, 224), validation_split=0.2, auto_augment=True,
                 cutout=True):
        train_val_datagen = ImageDataGenerator(horizontal_flip=True, validation_split=validation_split)
        test_datagen = ImageDataGenerator()

        self.train_iterator = train_val_datagen.flow_from_directory(directory=join(directory, 'train'),
                                                                    target_size=target_size,
                                                                    batch_size=batch_size,
                                                                    subset='training')

        self.validation_iterator = train_val_datagen.flow_from_directory(directory=join(directory, 'train'),
                                                                         target_size=target_size,
                                                                         batch_size=batch_size,
                                                                         subset='validation')

        self.test_iterator = test_datagen.flow_from_directory(directory=join(directory, 'val'),
                                                              target_size=target_size,
                                                              batch_size=batch_size,
                                                              )

        self.means = np.array(IMAGENET_MEANS)
        self.stds = np.array(IMAGENET_STDS)

        self.auto_augment = auto_augment
        self.cutout = cutout

        if auto_augment:
            self.policies = IMAGENET_POLICIES

    def _standardize(self, x):
        x = x.astype('float32') / 255

        means = self.means.reshape(1, 1, 1, 3)
        stds = self.stds.reshape(1, 1, 1, 3)

        x -= means
        x /= (stds + 1e-6)

        return x

    def _decolorize(self, x):
        channel_weights = np.random.uniform(low=0.2, high=0.6, size=(x.shape[0], 3))
        channel_weights = channel_weights / channel_weights.sum(axis=1)[:, None]
        channel_weights = channel_weights.reshape((x.shape[0], 1, 1, 3))
        x = (x * channel_weights).sum(axis=3).reshape((x.shape[0], x.shape[1], x.shape[2], 1))
        return x



    def _flow(self, iterator):

        while True:
            x_batch, y_batch = next(iterator)

            if self.cutout:
                for i in range(x_batch.shape[0]):
                    x_batch[i] = cutout(x_batch[i])

            if self.auto_augment:
                x_batch = x_batch.astype('uint8')
                for i in range(x_batch.shape[0]):
                    x_batch[i] = apply_policy(x_batch[i], self.policies[random.randrange(len(self.policies))])

            x_batch = self._standardize(x_batch)
            x_batch = self._decolorize(x_batch)

            yield x_batch, y_batch

    def train_flow(self):
        return self._flow(self.train_iterator)

    def val_flow(self):
        return self._flow(self.validation_iterator)

    def test_flow(self):
        return self._flow(self.test_iterator)






class Cifar10ImageDataGenerator:
    def __init__(self, auto_augment=True, cutout=True):
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, fill_mode='constant', cval=0,
                                          horizontal_flip=True)

        self.means = np.array([0.4914009, 0.48215896, 0.4465308])
        self.stds = np.array([0.24703279, 0.24348423, 0.26158753])

        self.auto_augment = auto_augment
        self.cutout = cutout

        if auto_augment:
            self.policies = CIFAR_10_POLICIES

    def standardize(self, x):
        x = x.astype('float32') / 255

        means = self.means.reshape(1, 1, 1, 3)
        stds = self.stds.reshape(1, 1, 1, 3)

        x -= means
        x /= (stds + 1e-6)

        return x

    def flow(self, x, y=None, batch_size=32, shuffle=True, sample_weight=None,
             seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None):
        batches = self.datagen.flow(x, y, batch_size, shuffle, sample_weight,
                                    seed, save_to_dir, save_prefix, save_format, subset)

        while True:
            x_batch, y_batch = next(batches)

            if self.cutout:
                for i in range(x_batch.shape[0]):
                    x_batch[i] = cutout(x_batch[i])

            if self.auto_augment:
                x_batch = x_batch.astype('uint8')
                for i in range(x_batch.shape[0]):
                    x_batch[i] = apply_policy(x_batch[i], self.policies[random.randrange(len(self.policies))])

            x_batch = self.standardize(x_batch)

            yield x_batch, y_batch


class Cifar100ImageDataGenerator:
    def __init__(self, auto_augment=True, cutout=True):
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, fill_mode='constant', cval=0,
                                          horizontal_flip=True)

        self.means = np.array([0.5071, 0.4867, 0.4408])
        self.stds = np.array([0.2675, 0.2565, 0.2761])

        self.auto_augment = auto_augment
        self.cutout = cutout

        if auto_augment:
            self.policies = CIFAR_10_POLICIES

    def standardize(self, x):
        x = x.astype('float32') / 255

        means = self.means.reshape(1, 1, 1, 3)
        stds = self.stds.reshape(1, 1, 1, 3)

        x -= means
        x /= (stds + 1e-6)

        return x

    def flow(self, x, y=None, batch_size=32, shuffle=True, sample_weight=None,
             seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None):
        batches = self.datagen.flow(x, y, batch_size, shuffle, sample_weight,
                                    seed, save_to_dir, save_prefix, save_format, subset)

        while True:
            x_batch, y_batch = next(batches)

            if self.cutout:
                for i in range(x_batch.shape[0]):
                    x_batch[i] = cutout(x_batch[i])

            if self.auto_augment:
                x_batch = x_batch.astype('uint8')
                for i in range(x_batch.shape[0]):
                    x_batch[i] = apply_policy(x_batch[i], self.policies[random.randrange(len(self.policies))])

            x_batch = self.standardize(x_batch)

            yield x_batch, y_batch


def cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    datagen = Cifar10ImageDataGenerator()
    x_test = datagen.standardize(x_test)
    # y_train = keras.utils.to_categorical(y_train, 10)
    # y_test = keras.utils.to_categorical(y_test, 10)

    # generator = ImageDataGenerator(
    #     rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest')
    #
    # generator.fit(x_train, seed=None, augment=False)

    return (x_train, y_train), (x_test, y_test), datagen


def cifar100_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    # x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    datagen = Cifar100ImageDataGenerator()
    x_test = datagen.standardize(x_test)
    # y_train = keras.utils.to_categorical(y_train, 10)
    # y_test = keras.utils.to_categorical(y_test, 10)

    # generator = ImageDataGenerator(
    #     rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest')
    #
    # generator.fit(x_train, seed=None, augment=False)

    return (x_train, y_train), (x_test, y_test), datagen
