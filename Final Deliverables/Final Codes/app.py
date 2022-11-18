import os
from flask import Flask,request,render_template,send_from_directory,url_for
#from flask_uploads import UploadSet, IMAGES,configure_uploads
# from flask_wtf import FlaskForm
# from flask_wtf.file import FileField,FileRequired,FileAllowed
from wtforms import SubmitField
from flask_cors import CORS
from keras_preprocessing import image
from keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np


app = Flask(__name__)
CORS(app)



model = load_model('final_model.h5')

user_list = ['ned','jon','arya','bran']
pass_list = ['head','snow','stark','wolf']

valid = ''
@app.route("/",methods=['GET','POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    print(username,password)



    for i in range(len(user_list)):
        if(username == user_list[i]):
            if(password == pass_list[i]):
                return render_template("lib.html")
            else:
                valid = 'invalid credentials'
                return render_template("Signin.html",valid=valid)
    valid = 'invalid credentials'
    return render_template("Signin.html",valid=valid)

@app.route("/Contact",methods=['GET','POST'])
def Contact():
    return  render_template("Contact.html")

@app.route("/About",methods=['GET','POST'])
def About():
    return  render_template("About.html")

@app.route("/predict",methods=['GET','POST'])
def predict():
    basepath = ''
    filepath = ''
    if request.method=='POST':
        f = request.files['image']
        print('current path')
        basepath = os.path.dirname(__file__)
        print("current path:",basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print('upload folder is',filepath)
        f.save(filepath)


    img = Image.open(filepath)
    img = img.resize((28,28))
    img = np.array(img)
    img = tf.image.rgb_to_grayscale(img)
    img = img.numpy()
    img = np.invert(img)
    img = img.astype("float32")
    img = img/255.0
    img = np.expand_dims(img,axis=0)
    pred = model.predict(img)
    prediction = np.argmax(pred)
    print("PREDICTION :",prediction)

    return render_template('lib.html',prediction=str(prediction))

file_url = None
app.run(debug=True)