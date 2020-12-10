from flask import Flask, request, jsonify
import json
import os
import time
import matplotlib
from tensorflow.python.keras.backend import global_learning_phase_is_set
matplotlib.use('agg')
from matplotlib import image
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import numpy as np
from shutil import copyfile
from train import CNN
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

USERNAME = "userName"
SDL = "sdl"
WIDTH = "width"
HEIGHT = "height"

SUCCESS_RESPONSE = {
    "status": "success"
}

ERROR_RESPONSE = {
    "status": "fail"
}

json_folder_path = "../json"
image_folder_path = "../image"

is_training = False

cnn = None

def currentTimeMilli() -> int:
    millis = int(round(time.time() * 1000))
    return millis

def saveJsonFile(username, jsonObject) -> str:
    # mkdir
    user_path = json_folder_path + "/" + username
    if not os.path.exists(json_folder_path):
        os.mkdir(json_folder_path)
    if not os.path.exists(user_path):
        os.mkdir(user_path)
    file_name = user_path + "/" + "swipe" + str(currentTimeMilli()) + ".json"
    with open(file_name, "w") as file:
        file.write(json.dumps(jsonObject))
    return file_name

def generatePredictImage(username: str, jsonContent, scale_factor: float) -> str:
    json_data = jsonContent
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
    fileName = "predict" + str(currentTimeMilli()) + ".jpg"
    plt.savefig(fileName, bbox_inches='tight')
    return fileName

def convertJsonFileToImage(username: str, filename: str, scale_factor: float):
    with open(filename,'r', encoding='utf8') as fp:
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
        imageDic = "../image"
        userImageDic = imageDic + "/" + username
        if not os.path.exists(imageDic):
            os.mkdir(imageDic)
        if not os.path.exists(userImageDic):
            os.mkdir(userImageDic)
        plt.savefig(userImageDic + "/" + os.path.splitext(filename)[0] + ".jpg", bbox_inches='tight')
        plt.close()

def moveImage(username: str, datasetDir: str):
    os.mkdir(datasetDir)
    os.mkdir(os.path.join(datasetDir, username))
    os.mkdir(os.path.join(datasetDir, "others"))
    # iterate
    dir_list = os.listdir(image_folder_path)
    for dir in dir_list:
        cur_path = os.path.join(image_folder_path, dir)
        if os.path.isdir(cur_path):
            print("current path:", cur_path)
            if dir == username:
                # me
                #copy to  dataset
                for file in os.listdir(cur_path):
                    copyfile(os.path.join(cur_path, file), os.path.join(datasetDir, username, file))
            else:
                #others
                for file in os.listdir(cur_path):
                    copyfile(os.path.join(cur_path, file), os.path.join(datasetDir, "others", file))

def train(cnn: CNN):
    cnn.train()

@app.route('/senddata', methods=['POST'])
def senddata():
    content = request.json
    print("/senddata:", "received ")
    if USERNAME in content and SDL in content and HEIGHT in content and WIDTH in content and len(content) == 4:
        filename = saveJsonFile(content[USERNAME], content[SDL])
        convertJsonFileToImage(content[USERNAME], filename, 960/content[HEIGHT])
        return jsonify(SUCCESS_RESPONSE)
    else:
        return jsonify(ERROR_RESPONSE), 400

@app.route('/train', methods=['POST'])
def starttrain():
    content = request.json
    if USERNAME in content and len(content) == 1 and os.path.exists(json_folder_path+"/"+content[USERNAME]):
        datasetDir = "../dataset" + str(currentTimeMilli())
        moveImage(content[USERNAME], datasetDir=datasetDir)
        global cnn
        cnn = CNN(datasetDir, content[USERNAME])
        executor = ThreadPoolExecutor()
        executor.submit(train, cnn)
        return jsonify(SUCCESS_RESPONSE)
    else:
        return jsonify(ERROR_RESPONSE), 400

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    if USERNAME in content and SDL in content and HEIGHT in content and WIDTH in content and len(content) == 4:
        filename = generatePredictImage(content[USERNAME], content[SDL], 960/content[HEIGHT])
        prediction = cnn.predict(filename)
        print(prediction)
        os.remove(filename)
        return jsonify(SUCCESS_RESPONSE)
    else:
        return jsonify(ERROR_RESPONSE), 400

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5002)