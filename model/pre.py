import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
import numpy as np
model=keras.models.load_model('weight2.h5')
print(model.inputs)
filename = 'anhui23.jpg'
#img = cv2.imread(filename)

#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#res=cv2.resize(img,(200,200),interpolation=cv2.INTER_LINEAR)
img = image.load_img(filename, grayscale=False,target_size=(100, 200))

#x = image.img_to_array(img)
#print(x)
#x = np.expand_dims(x, axis=0)
img = np.expand_dims(img, axis=0)
#res = np.expand_dims(res, axis=3)
#print(x.shape)
preds = model.predict(img)
print(preds)
