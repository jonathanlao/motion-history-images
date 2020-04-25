import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None


def show(img, wait=0, expand=False):  # helpful for debugging
    tmp = img.copy()
    if expand:
        tmp = cv2.resize(tmp, (500, 500))
    cv2.imshow('image', normalize(tmp))
    cv2.waitKey(wait)


def normalize(img):
    return cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def write(file, images, image_index):
    cv2.imwrite(file, normalize(images[image_index]))


def generate_images(type, action, images):
    file = type + '_' + action + '.png'
    if type == 'binary':
        if action == 'handwaving':
            write(file, images, 7)
        elif action == 'boxing':
            write(file, images, 4)
        elif action == 'handclapping':
            write(file, images, 6)
        elif action == 'walking':
            write(file, images, 31)
        elif action == 'jogging':
            write(file, images, 27)
        elif action == 'running':
            write(file, images, 11)
    elif type == 'mhi':
        write(file, images, 20)


def mp4_video_writer(filename, frame_size, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def cross_validate(x_train, y_train, x_test, y_test):  # using grid search instead for now
    n_neighbors = 2
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_k = None
    best_score = 0
    for train_index, test_index in kf.split(x_train):
        x_train_folds, x_val_fold = x_train[train_index], x_train[test_index]
        y_train_folds, y_val_fold = y_train[train_index], y_train[test_index]
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(x_train_folds, y_train_folds)
        score = knn.score(x_val_fold, y_val_fold)
        print('K = ', n_neighbors, ':', score)
        if score > best_score:
            best_k = n_neighbors
        n_neighbors += 1

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    final_score = knn.score(x_test, y_test)
    matrix = confusion_matrix(y_test, pred)
    # print(final_score)
    # print(matrix)
