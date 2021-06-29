import keras
import cv2
import numpy as np

model = keras.models.load_model('model1.h5')

def predict(image):
    
    img = cv2.resize(image, dsize = (128, 128))
    #img = image/ 255
    cv2.imwrite("pp.jpg",img)
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    probs = model.predict(img.reshape(1, 128, 128, 3))[0]
    print("Predicted: "+ str(np.argmax(probs)) + " with "+  str(round(np.max(probs)*100)) + "% accuracy")

