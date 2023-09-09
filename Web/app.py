from flask import Flask, render_template, request
from keras.models import load_model
from numpy import argmax
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import pandas as pd

app = Flask(__name__)

dic_1 = {0:'Other', 1:'Mango'}
dic_2 = {0:'Rotten mango', 1:'Raw mango', 2:'Ripe mango'}

model_1 = load_model('../Web/model/m1.h5')
model_1.make_predict_function()

model_2 = load_model('../Web/model/model_x.h5')
model_2.make_predict_function()

#pred
def pred_model(img_path, model):
    #load image
    # img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
    # img_array = tf.keras.utils.img_to_array(img)/255.0
    # img_pred = img_array.reshape(1, 128, 128, 3)
    #predict
    # p = model.predict(img_pred)
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)
    imgBatch = np.expand_dims(img_array, axis=0)
    imgPreprocessed = preprocess_input(imgBatch)
    p = model.predict(imgPreprocessed)

    return p

def res_cvt(result):
    res = argmax(result, axis=1)
    return res

def acc_result(result, res):
    acc = round(result[0][res[0]]*100,2)
    acc = str(acc)
    return acc

def lable_result(dic, res):
    lable = dic[res[0]]
    return lable

#routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template('index.html')

@app.route("/about")
def about_page():
    return "TRAMNQ"

@app.route('/submit', methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = 'static/img/' + img.filename
        img.save(img_path)       
        #pred 1
        pred = pred_model(img_path, model_1)
        lable = lable_result(dic_1, res_cvt(pred))
        print("Ket qua mo hinh 1:", pred)
        if(lable == 'Mango'):
            pred = pred_model(img_path, model_2)
            lable = lable_result(dic_2, res_cvt(pred))
            acc = acc_result(pred, res_cvt(pred))
            print("Ket qua mo hinh 2:", pred)
            return render_template('index.html', prediction = "Mango", status = lable + " ({}%)".format(acc),img_path = img_path)
        else:
            acc = acc_result(pred, res_cvt(pred))
        result = lable + " ({}%)".format(acc)
        print(result + "\n")
    return render_template('index.html', prediction = result, status = "N/A" ,img_path = img_path)

if __name__ == '__main__':
    app.run(debug=True)