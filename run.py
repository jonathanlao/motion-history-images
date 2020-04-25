import cv2
import numpy as np
import helper
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import sys


def preprocess_frame(frame, k, sigma):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, k, sigma)
    return frame


def binary_motion_signal(video_file, theta=20, debug=False, skip_motionless_frames=True):
    blur_ksize = (3,) * 2
    blur_sigma = 0

    frame_gen = helper.video_frame_generator(video_file)
    frame = frame_gen.__next__()

    frame = preprocess_frame(frame, blur_ksize, blur_sigma)

    binary_motion_signal_images = []
    next_frame = frame_gen.__next__()
    while next_frame is not None:
        next_frame = preprocess_frame(next_frame, blur_ksize, blur_sigma)

        image_diff = np.abs(cv2.subtract(frame, next_frame))
        image_diff[image_diff < theta] = 0
        image_diff[image_diff >= theta] = 1
        binary_image = image_diff

        if not skip_motionless_frames or binary_image.sum() != 0:  # skip motionless frames
            binary_motion_signal_images.append(binary_image)

            if debug:
                helper.show(binary_image, 20)

        frame = next_frame
        next_frame = frame_gen.__next__()

    return binary_motion_signal_images


def get_motion_history_images(binary_motion_signal_images, max_value):
    mhis = []
    img = binary_motion_signal_images[0].astype(np.float32)
    img[img > 0] = max_value
    mhis.append(img)

    mhi = img
    for next_img in binary_motion_signal_images[1:]:
        next_img = next_img.astype(np.float32)
        next_img[next_img > 0] = max_value
        mhi = mhi - 1

        mhi[mhi < 0] = 0

        mhi = next_img + mhi
        mhi[mhi > max_value] = max_value

        mhis.append(mhi)

    return mhis


def get_central_moment(image, i, j, x_mean=0, y_mean=0):
    x = (np.arange(image.shape[1]) - x_mean) ** i
    y = ((np.arange(image.shape[0]) - y_mean) ** j).reshape((-1, 1))
    moment = np.sum(x * y * image)
    return moment


def hu_moments(image, pq):
    if image.sum() == 0:
        return [0] * len(pq)

    m00 = get_central_moment(image, 0, 0)
    m01 = get_central_moment(image, 0, 1)
    m10 = get_central_moment(image, 1, 0)

    x_mean = m10 / m00
    y_mean = m01 / m00

    scale_invariant_moments = []
    u00 = get_central_moment(image, 0, 0, x_mean, y_mean)
    for val in pq:
        hu_moment = get_central_moment(image, val[0], val[1], x_mean, y_mean)

        # use scale invariant instead of central moment
        exp = 1 + ((val[0] + val[1]) / 2)
        scale_invariant_moment = hu_moment / (np.power(u00, exp))
        scale_invariant_moments.append(scale_invariant_moment)
    return scale_invariant_moments


def find_best_k(x_train, y_train):
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 10)}
    knn_gscv = GridSearchCV(knn, param_grid, cv=5)
    knn_gscv.fit(x_train, y_train)
    return knn_gscv.best_params_['n_neighbors']


def get_data():
    # Define Parameters
    theta = 20
    tau = 20

    scale_invariant_moments = []
    labels = []
    thetas = [theta] * 6
    taus = [tau] * 6
    classes = ['handwaving', 'boxing', 'handclapping', 'walking', 'jogging', 'running']

    for theta, tau, action in zip(thetas, taus, classes):
        binary_motion_signal_images = []
        for i in range(1, 10):
            for j in range(1, 3):
                person = str(i)
                if i < 10:
                    person = '0' + person

                file = 'input/' + action + '/person' + person + '_' + action + '_d' + str(j) + '_uncomp.avi'
                images = binary_motion_signal(file, theta=theta, debug=False)
                binary_motion_signal_images.extend(images)

        helper.generate_images('binary', action, binary_motion_signal_images)

        mhis = get_motion_history_images(binary_motion_signal_images, tau)

        debug = False
        if debug:
            for j in mhis:
                helper.show(j, 0, True)

        helper.generate_images('mhi', action, mhis)

        pq = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (2, 2)]
        for mhi in mhis:
            hu = hu_moments(mhi, pq)
            scale_invariant_moments.append(hu)
            labels.append(action)

    scale_invariant_moments = np.asarray(scale_invariant_moments)
    labels = np.asarray(labels)

    return scale_invariant_moments, labels


if __name__ == '__main__':
    GET_DATA = False
    if GET_DATA:
        data, labels = get_data()
        data = np.save('data.npy', data)
        labels = np.save('labels.npy', labels)

    data = np.load('data.npy')
    labels = np.load('labels.npy')

    random_state = 42
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=random_state)

    TRAIN_MODEL = False
    k = 1
    if TRAIN_MODEL:
        k = find_best_k(x_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    GENERATE_REPORT = False
    if GENERATE_REPORT:
        pred = knn.predict(x_test)
        final_score = knn.score(x_test, y_test)
        matrix = confusion_matrix(y_test, pred)
        # print(final_score)
        titles_options = [("Confusion matrix, without normalization", 'matrix.png', None, '.5g'),
                          ("Normalized confusion matrix", 'matrix_normalized.png', 'true', '.2g')]
        for title, file, normalize, values_format in titles_options:
            disp = plot_confusion_matrix(knn, x_test, y_test,
                                         display_labels=['waving', 'boxing', 'clapping', 'walking', 'jogging', 'running'],
                                         cmap=plt.cm.Blues,
                                         values_format=values_format,
                                         normalize=normalize)
            plt.title(title)
            plt.savefig(file)

    GENERATE_IMAGE = False
    if len(sys.argv) == 2:
        input_video = sys.argv[1]
        theta = 20
        tau = 20
        binary_images = binary_motion_signal(input_video, theta=theta, debug=False, skip_motionless_frames=False)
        mhis = get_motion_history_images(binary_images, tau)
        pq = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (2, 2)]
        moments = []
        for mhi in mhis:
            hu = hu_moments(mhi, pq)
            moments.append(hu)

        pred = knn.predict(moments)

        # Ideally, I should calculate the predictions and output the video in 1 pass instead of 2
        frame_gen = helper.video_frame_generator(input_video)
        frame = frame_gen.__next__()
        h, w, d = frame.shape
        fps = 40

        video_out = helper.mp4_video_writer('output.mp4', (w, h), fps)

        frame_index = 0
        while frame is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, h-10)
            fontScale = 0.5
            fontColor = (0, 0, 255)
            lineType = 2

            # Since mhis are calculated by diffs of images, there will
            # necessarily be one less prediction than the number of frames
            img = frame
            if frame_index != 0 and binary_images[frame_index-1].sum() != 0:
                img = cv2.putText(frame, pred[frame_index-1],
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

            if GENERATE_IMAGE and frame_index == 100:
                cv2.imwrite('classify1_handwaving.png', helper.normalize(img))
            if GENERATE_IMAGE and frame_index == 110:
                cv2.imwrite('classify2_handwaving.png', helper.normalize(img))
            if GENERATE_IMAGE and frame_index == 148:
                cv2.imwrite('classify_handclapping.png', helper.normalize(img))

            video_out.write(img)
            frame = frame_gen.__next__()
            frame_index += 1

        video_out.release()
