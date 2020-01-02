#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import model_from_json


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# In[3]:


vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
epsilon = 0.40

#you can download the pretrained weights from the following link 
#https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
#or you can find the detailed documentation https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/


#model.load_weights('C:/Users/pcc14/Downloads/tensorflow-101-master/python/vgg_face_weights.h5')


# In[4]:


class  Face():
	"""docstring for  text"""

	def __init__(self):
		self.model = model.load_weights('C:/Users/pcc14/Downloads/tensorflow-101-master/python/vgg_face_weights.h5')

	def preprocess_image(self, image_path):
	    img = load_img(image_path, target_size=(224, 224))
	    img = img_to_array(img)
	    img = np.expand_dims(img, axis=0)
	    img = preprocess_input(img)
	    return img


	def findCosineSimilarity(self, source_representation, test_representation):
	    a = np.matmul(np.transpose(source_representation), test_representation)
	    b = np.sum(np.multiply(source_representation, source_representation))
	    c = np.sum(np.multiply(test_representation, test_representation))
	    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

	def findEuclideanDistance(self, source_representation, test_representation):
	    euclidean_distance = source_representation - test_representation
	    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
	    euclidean_distance = np.sqrt(euclidean_distance)
	    return euclidean_distance




	def verifyFace(self, img1, img2):

	    #img1_representation = vgg_face_descriptor.predict(self.preprocess_image('C:/Users/pcc14/Downloads/tensorflow-101-master/python/trainset/%s' % (img1)))[0,:]
	    #img2_representation = vgg_face_descriptor.predict(self.preprocess_image('C:/Users/pcc14/Downloads/tensorflow-101-master/python/trainset/%s' % (img2)))[0,:]
	    img1_representation = vgg_face_descriptor.predict(self.preprocess_image(img1))[0,:]
	    img2_representation = vgg_face_descriptor.predict(self.preprocess_image(img2))[0,:]
	        
	    
	    cosine_similarity = self.findCosineSimilarity(img1_representation, img2_representation)
	    euclidean_distance = self.findEuclideanDistance(img1_representation, img2_representation)
	    
	    #print("Cosine similarity: ",cosine_similarity)
	    #print("Euclidean distance: ",euclidean_distance)
	    

	    if(cosine_similarity < epsilon):
	        #print("verified... they are same person")
	        #return cosine_similarity
	        return 'verified... they are same person'
	        
	    else:
	        #print("unverified! they are not same person!")
	        #return cosine_similarity
	        return 'unverified! they are not same person!'

	    # f = plt.figure()
	    # f.add_subplot(1,2, 1)
	    # plt.imshow(image.load_img('C:/Users/pcc14/Downloads/tensorflow-101-master/python/trainset/%s' % (img1)))
	    # plt.xticks([]); plt.yticks([])
	    # f.add_subplot(1,2, 2)
	    # plt.imshow(image.load_img('C:/Users/pcc14/Downloads/tensorflow-101-master/python/trainset/%s' % (img2)))
	    # plt.xticks([]); plt.yticks([])
	    # plt.show(block=True)
	    # print("-----------------------------------------")


# In[ ]:

# In[6]:






my_face=Face()


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = "super secret key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/faceCompare', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if 'file1' not in request.files:
        	flash('No file')
        	return redirect(request.url)
        file1 = request.files['file1']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file1.filename == '':
        	flash('No selected file')
        	return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename1 = secure_filename(file1.filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))

            return  my_face.verifyFace("uploads/"+filename,"uploads/"+filename1)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file><br>
        <input type=file name=file1>
      <input type=submit value=Upload>
    </form>
    </form>
    '''


if __name__ == '__main__':

    app.run(host='0.0.0.0',port=8080, threaded=False, debug=False)




