from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import argparse as ap
from nms import nms
import sys



orientations = 9
pixels_per_cell = [8, 8]
cells_per_block = [3, 3]
visualize = False
normalize = True

def sliding_window(image, window_size, step_size):

    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])



im = imread(str(sys.argv[1]), as_grey=False)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)    
min_wdw_sz = (128, 128)
step_size = (10, 10)
downscale = 1.2
visualize_det = False

clf = joblib.load("./svm model/car_svm.model")

detections = []
scale = 0

for im_scaled in pyramid_gaussian(im, downscale=downscale):

    cd = []

    if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
        break
    for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
        if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
            continue

        fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        pred = clf.predict(fd)
        if pred == 1:
            # print  "Detection:: Location -> ({}, {})".format(x, y)
            # print "Scale ->  {} | Confidence Score {} \n".format(scale,clf.decision_function(fd))
            detections.append((x, y, clf.decision_function(fd),
                int(min_wdw_sz[0]*(downscale**scale)),
                int(min_wdw_sz[1]*(downscale**scale))))
            cd.append(detections[-1])

        if visualize_det:
            clone = im_scaled.copy()
            for x1, y1, _, _, _  in cd:
                # Draw the detections at this scale
                cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                    im_window.shape[0]), (0, 0, 255), thickness=2)
            cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                im_window.shape[0]), (0, 0, 255), thickness=2)
            cv2.imshow("Sliding Window in Progress", clone)
            cv2.waitKey(30)

    scale+=1

clone = im.copy()
# for (x_tl, y_tl, _, w, h) in detections:
#     # Draw the detections
#     cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 0, 0), thickness=2)
# cv2.imshow("Raw Detections before NMS", im)
# cv2.waitKey()


detections = nms(detections, 0.3)

for (x_tl, y_tl, _, w, h) in detections:

    cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 255, 0), thickness=2)
cv2.imshow("Final Detections after applying NMS", clone)
cv2.waitKey()
