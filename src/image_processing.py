import glob
import json
import os
from matplotlib import image
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import numpy as np

def generate_image_from_json(filename: str, output_path: str, scale_factor: float):
    with open(filename,'r',encoding='utf8') as fp:
        json_data = json.load(fp)
        x_points = np.array([one['x']*scale_factor for one in json_data])
        y_points = np.array([one['y']*scale_factor for one in json_data])
        plt.figure(figsize=(9, 16))
        plt.xlim(0.0, 540.0)
        plt.ylim(0.0, 960.0)
        plt.plot(x_points, y_points)

        # plt.scatter(x_points, y_points, s=1**2)

        # hide axis
        plt.axis('off')
        # remove white padding
        filename = filename.split('/')[-1]
        plt.savefig(output_path + "/" + os.path.splitext(filename)[0] + ".jpg", bbox_inches='tight')
        plt.close()

JSON_PATH = "json_path"
IMAGE_PATH = "image_path"
IS_TRUE = "is_true"
SCALE_FACTOR = "scale_factor"

image_folder_path = "../image"

configuration = {
    "jianan": {
        JSON_PATH: r"../json/swipedata-jianan/*.json",
        IMAGE_PATH: "../image/jianan",
        SCALE_FACTOR: 0.6667,
        IS_TRUE: True
    },
    "cheng": {
        JSON_PATH: r"../json/swipedata-cheng/*.json",
        IMAGE_PATH: "../image/cheng",
        SCALE_FACTOR: 1.0,
        IS_TRUE: False
    },
    "flei": {
        JSON_PATH: r"../json/swipedata-flei/*.json",
        IMAGE_PATH: "../image/flei",
        SCALE_FACTOR: 1.0,
        IS_TRUE: False
    },
    "wei": {
        JSON_PATH: r"../json/swipedata-wei/*.json",
        IMAGE_PATH: "../image/wei",
        SCALE_FACTOR: 1.0,
        IS_TRUE: False
    },
    "mc": {
        JSON_PATH: r"../json/swipedata-mc/*.json",
        IMAGE_PATH: "../image/mc",
        SCALE_FACTOR: 0.6667,
        IS_TRUE: False
    }
}

# mkdir
if not os.path.exists(image_folder_path):
    os.mkdir(image_folder_path)
for name, value in configuration.items():
    image_path = value[IMAGE_PATH]
    if not os.path.exists(image_path):
        os.mkdir(image_path)

# generate images
for name, value in configuration.items():
    fileList = glob.glob(value[JSON_PATH])
    for file in fileList:
        generate_image_from_json(file, value[IMAGE_PATH], value[SCALE_FACTOR])
