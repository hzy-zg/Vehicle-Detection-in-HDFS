from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import os

features_positive = "./features/positive"
features_negative = "./features/negative"

fds = []
labels = []

count = 1
for features in glob.glob(os.path.join(features_positive,"*.feat")):
    fd = joblib.load(features)
    fds.append(fd)
    if count < 10:
        print len(fd)
        print '\n'
        print len(fds)
        print '\n'
        count+=1
        if count == 9:
            print "-------------------\n\n"
    labels.append(1)

for features in glob.glob(os.path.join(features_negative,"*.feat")):
    fd = joblib.load(features)
    fds.append(fd)
    if count < 20:
        print len(fd)
        print '\n'
        print len(fds)
        print '\n'
        count+=1
        
    labels.append(0)

clf = LinearSVC()
print "SVM Training"
clf.fit(fds, labels)

classifier_path = './model/test_car_svm.model'

joblib.dump(clf, classifier_path)
print "SVM Classifier Saved"
