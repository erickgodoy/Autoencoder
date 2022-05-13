from flask import Flask, render_template, request, redirect, url_for
import requests 

import cv2
import tensorflow as tf
import os
import tkinter as tk
import numpy as np 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.preprocessing import image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from tensorflow import keras


app = Flask(__name__)


UPLOAD_FOLDER = 'static/images'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def result():
    return render_template('result.html')


@app.route('/')
def pixalate_image(image, scale_percent = 40):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
               
    width = int(small_image.shape[1] * 100 / scale_percent)
    height = int(small_image.shape[0] * 100 / scale_percent)
    dim = (width, height)
    low_res_image = cv2.resize(small_image, dim, interpolation =  cv2.INTER_AREA)
    return low_res_image

@app.route('/result', methods=['POST'])
def back_page():
    filename1 = 'image.jpg'
    os.unlink(os.path.join(app.config('UPLOAD_FOLDER'), filename1))

    filename2 = 'image.jpg'
    os.unlink(os.path.join(app.config('UPLOAD_FOLDER'), filename2))
    
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    name = uploaded_file.filename

    img = image.load_img('imagenes\\'+name,  target_size=(80,80,3))
    img = image.img_to_array(img)
    img = img/255

    img = pixalate_image(img)
    
    plt.imshow(img)
    plt.savefig("static\\images\\image.jpg")
    
    input_array = np.array([img])

    autoencoder =  keras.models.load_model("model_rn.h5")
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    predict = autoencoder.predict(input_array)

    plt.imshow(predict[0])
    plt.savefig("static\\images\\image2.jpg")

    return render_template("result.html")



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)

