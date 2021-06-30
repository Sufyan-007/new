import keras
import cv2
import numpy as np
from list import C
# list of celebrities name vs index
model = keras.models.load_model('model1.h5')

def predict(image):
    # processing image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, dsize = (128, 128))
    #predicting result, x is the index value of the celeb
    res=model.predict(image.reshape(1, 128, 128, 3))[0]
    x=np.argmax(res)
    #importing result picture
    p= 'pics\\'
    p=p+str(x)+'.jpg'
    img = cv2.imread(p)
    img = cv2.resize(img, (1000,800))
    #adding captions and saving picture to display
    img=cv2.copyMakeBorder(img,100, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(img,"YOUR CELEBRITY LOOKALIKE",(10,50), cv2.FONT_HERSHEY_COMPLEX,2,(0,255,255),2)
    cv2.putText(img,C[x].upper(),(50,800), cv2.FONT_HERSHEY_COMPLEX,2.75,(255,255,0),2)
    cv2.imwrite("predicted.jpg",img)

