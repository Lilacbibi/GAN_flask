from flask import Flask, render_template, request, redirect, url_for, flash, Response
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64

app = Flask(__name__)
app.secret_key = 'b782b926aefe9fee1115ecfbb6d8f3bcf5b8981cbb90d89a'

#Constants
num_classes = 3
latent_dim =128
list = ['anime', 'cartoon', 'human']


print('333333333333333333333333333333333333333333')
@app.route('/')
def index():
    return render_template('submission.html')

@app.route('/prediction', methods=('GET','POST'))    
def gan():
    #Generate noise with input of label number
    class_num = int(request.form['LabelNumber'])
    label_name = list[class_num]
    one_hot_labels = keras.utils.to_categorical(class_num, num_classes)[None,:]
    random_latent_vectors = tf.random.normal(shape=(1, latent_dim))
    random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1)
    #Load the model and generate image with noise
    gan_model = keras.models.load_model('cGAN.h5', compile=False)
    print('--------------------------------------------')
    image = gan_model.predict(random_vector_labels, verbose=0)
    img = (image * 127.5 + 127.5).astype(np.uint8)

    #Plot the image
    """fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.imshow(np.squeeze(img))
    ax.axis('tight')
    ax.axis('off')
    pngImage = BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')"""
    return render_template('/prediction.html', image='', label=label_name)

if(__name__ == "__main__"):
    app.run(debug=True)