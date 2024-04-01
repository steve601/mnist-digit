from flask import Flask,render_template,request
import numpy as np
from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model
import os

app = Flask(__name__)
model = load_model('mnhnd.keras')


UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def homepage():
    return render_template('mnist.html')

@app.route("/detect",methods = ['POST'])
def recognize():
    imgfile = request.files['digit']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imgfile.filename)
    imgfile.save(image_path)
    
    img = load_img(image_path)
    img = img.resize((28, 28))
    img_arr = img_to_array(img)/ 255.0
    img_arr = img_arr.reshape(-1, 28, 28, 1)
    
    pred = model.predict(img_arr)
    label = np.argmax(pred,axis=1)[0]
    
    text = f'This is digit {label}'
    
    return render_template('mnist.html',pred = text,img_path = image_path)

if __name__ == '__main__':
    app.run(debug=True)