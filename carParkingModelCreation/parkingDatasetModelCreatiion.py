import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# prepare data
input_dir = "/home/golu/PycharmProjects/imageClassification/clf-data"
categories = ["empty", "not_empty"]

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)
        # we are not going append a image, but we are  going to append an array and img flatten will do this work

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)
# wea re creating two array from a data exsting as data ,so we are doiung this is as test_size = 0.2 means we are
# allocating 20% of thw data we have fro testing adn rest 80% will be going for the training shuffle is very
# important to make the data shuffl;e so that we can create a well distributred data stratify  is used to ensure that
# we have all the label s in the same proportionas we have in the dataset so that we ensurea well uniform creaeting
# of test and training

# train classifier
classifier = SVC()

parameters = [{"gamma": [0.01, 0.001, 0.0001], "C": [1, 10, 100, 1000]}]
# we are actually going to train multiple classifier for each combination of c and gamma, so we are going to have 12
# classifier
grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print("{}% of samples were correctly classified".format(str(score * 100)))

pickle.dump(best_estimator, open("./model.p", "wb"))
