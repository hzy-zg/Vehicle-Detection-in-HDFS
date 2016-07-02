import cv2
import sys
import cv
import numpy as np
from datetime import datetime

def mapper(key, value):

    imgbytes = np.fromstring(value,dtype='uint8') 
    imarr = cv2.imdecode(imgbytes,cv2.CV_LOAD_IMAGE_COLOR) 
    # img = cv.fromarray(imarr) 
    # im = numpy.array(im)


    cv_img = imarr.astype(np.uint8)
    cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    cascade_src = 'cars.xml'

    car_cascade = cv2.CascadeClassifier(cascade_src)

    cars = car_cascade.detectMultiScale(cv_gray, 1.01, 1)

    no_cars = int(len(cars))
    # Yield count of colour... 
    yield str(datetime.now()), no_cars 

def reducer(key, values):
    # Simply sum up the values per colour...
    yield key, sum(values)

if __name__ == "__main__":
    import dumbo
    dumbo.run(mapper, reducer, combiner=reducer)
