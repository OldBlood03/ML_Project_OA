from os import listdir
from os import getcwd
from os.path import isfile, splitext
import numpy as np
import keras
import cv2
from matplotlib.pyplot import imread


def read_data_from_files(
    *fpaths,
):  # read data from all files in the listed directories *fpaths and process the images into (np.array, str) tuples
    cwd = "c:\\Users\\Macabre\\OneDrive\\Työpöytä\\ML_Project_OA\\src\\"
    return (
        (imread(cwd + "{}\\{}".format(dir, fil)), dir)
        for dir in fpaths
        for fil in listdir(cwd + dir)
        if isfile(cwd + "{}\\{}".format(dir, fil))
    )


def preprocess_data(datagenerator, xscale, yscale, grey=True):
    if grey:
        return (
            (inetrpolate(greyscale(feature), xscale, yscale), label)
            for (feature, label) in datagenerator
        )
    else:
        return (
            (inetrpolate(feature, xscale, yscale), label)
            for (feature, label) in datagenerator
        )


def inetrpolate(arr, *scale):
    return cv2.resize(arr, dsize=scale, interpolation=cv2.INTER_CUBIC)


def greyscale(img):  # takes uint8 image tensor
    # The RGB values are converted to grayscale using the NTSC formula: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
    # assuming the colors listed in the image tensor img of shape (x,y,3) are in rgb order
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114


def segment_data(generator, trainingpercent):
    shuffled_arr = np.random.shuffle(generator.copy())
    return (
        shuffled_arr[: int(trainingpercent * len(generator)), :, :],
        shuffled_arr[int(trainingpercent * len(generator)) :, :, :],
    )


beefile = "Bees"
notbeefile = "Not_Bees"

from PIL import Image

data = read_data_from_files(notbeefile)
processed_data = preprocess_data(data, 300, 300, False)
# (training_data,testing_data) = segment_data(processeddata)
while input("type q to quit: ") != "q":
    datum = next(processed_data)[0]
    print(datum.shape)
    Image.fromarray(np.uint8(datum * 255)).show()
