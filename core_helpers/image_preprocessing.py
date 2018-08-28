from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import zscore
import os
from shutil import rmtree


FLAGS = tf.app.flags.FLAGS


def normalize_landmarks(landmarks):
    return zscore(landmarks, axis=0)


def resize_landmarks(landmarks, old_size, new_size, axis):
    """
    Resizes the landmark coordinates of a resized image to fit in.
    :param landmarks: The landmarks at the ground truth as a list.
    :param old_size: Old size of the image(axis 0 or 1 )
    :param new_size: New size of the image(axis 0 or 1)
    :param axis: You know what
    :return: resized landmarks as a list.
    """
    # Define major coefficient, happen to be found by experiments
    # it decreases log-rate when image size get smaller and vice versa
    # Hint: when new_size is 100 this value works best with 180~and 72 for 40
    major_coefficient = 72
    if axis == 0:
        """ Change in y axis"""
        new_coords = [val-new_size/major_coefficient for val in (np.array(landmarks[5:10]) * new_size / old_size)]
        new_landmarks = landmarks[0:5] + new_coords
    elif axis == 1:
        """ Change in x axis"""
        new_coords = [val-new_size/major_coefficient for val in (np.array(landmarks[0:5]) * new_size / old_size)]
        new_landmarks = new_coords + landmarks[5:10]
    return new_landmarks


def rotate_landmarks(landmarks, theta):
    """
        Rotated landmarks around given theta
        :param landmarks:
        :param theta:
        :return:
    """
    rotated_landmarks = []
    # Rotation matrix to rotate landmarks:
    radian = np.radians(theta)
    cos, sin = np.cos(radian), np.sin(radian)
    R = np.matrix([[cos, -sin], [sin, cos]])
    # Rotate landmarks
    rotated_landmarks.extend([l[0] for l in np.reshape(np.dot(np.reshape(landmarks, (5, 2)), R.T), (10, 1)).tolist()])
    return rotated_landmarks


def uniform_aspect_ratio(dataset):
    """

        :param dataset:
        :return: new dataset= {"images": PIL image, "landmarks": list, "poses": list, "fnames": list}
    """
    train_image_path = dataset["train"]["file_path"]
    train_landmarks = dataset["train"]["landmarks"]
    train_poses = dataset["train"]["poses"]

    test_image_path = dataset["test"]["file_path"]
    test_landmarks = dataset["test"]["landmarks"]
    test_poses = dataset["test"]["poses"]

    train_size = len(train_image_path)

    f_data_set = {"file_path": [], "image": [], "landmark": [], "pose": []}

    image_paths = train_image_path + test_image_path
    landmarks = train_landmarks + test_landmarks
    poses = train_poses + test_poses

    for i, image_path in enumerate(image_paths):
        f_data_set["file_path"].append(image_path)
        image = Image.open(image_path)

        width = landmarks[i][1] - landmarks[i][0]
        height = landmarks[i][8] - landmarks[i][5]
        aspect_ratio = width/height

        if aspect_ratio < 1:
            # print("Height is bigger")
            new_height = int(image.height*aspect_ratio)
            resized_image = image.resize((image.width, new_height))
            rszd_landmarks = resize_landmarks(landmarks[i], image.height, new_height, axis=0)
        elif aspect_ratio > 1:
            # print("Width is bigger")
            new_width = int(image.width*aspect_ratio)
            resized_image = image.resize((new_width, image.height))
            rszd_landmarks = resize_landmarks(landmarks[i], image.width, new_width, axis=1)

        f_data_set["image"].append(resized_image)
        f_data_set["landmark"].append(rszd_landmarks)
        f_data_set["pose"].append(poses[i])

        # Log new aspect ratio
        # ner = (rszd_landmarks[1]-rszd_landmarks[0]) / (rszd_landmarks[8]-rszd_landmarks[5])
        # print("New aspect ratio is "+str(ner))
        # plt.imshow(np.asarray(resized_image), cmap='gray')
        # plt.plot(landmarks[i][0:5], landmarks[i][5:10], 'r.')
        # plt.plot(rszd_landmarks[0:5], rszd_landmarks[5:10], 'g.')
        # plt.show()
    # Train test split
    train_data = {"file_path": f_data_set["file_path"][0:train_size],
                  "image": f_data_set["image"][0:train_size],
                  "landmark": f_data_set["landmark"][0:train_size],
                  "pose": f_data_set["pose"][0:train_size]}
    test_data = {"file_path": f_data_set["file_path"][train_size:],
                 "image": f_data_set["image"][train_size:],
                 "landmark": f_data_set["landmark"][train_size:],
                 "pose": f_data_set["pose"][train_size:]}

    return {"train": train_data, "test": test_data}


def force_to_resize(data_set):
    """
        Force te resize to make all faces in the same size through the dataset.
        :param data_set: is an input what do you want me to line here.
        All images(faces) in the dataset must be in the same scale for that,
        this function loops through all dataset
        finds the mean face window size
        trims down or pads up all faces to exhibit same size.
        :return: r_data_set: Resized dataset well thats a shock.
    """

    images = data_set["train"]["image"] + data_set["test"]["image"]
    landmarks = data_set["train"]["landmark"] + data_set["test"]["landmark"]

    train_size = len(data_set["train"]["image"])

    widths = []
    heights = []
    for i in range(len(landmarks)):
        widths.append(landmarks[i][1] - landmarks[i][0])
        heights.append(landmarks[i][8] - landmarks[i][5])
    mean_width = int(np.mean(np.array(widths), axis=0))
    mean_height = int(np.mean(np.array(heights), axis=0))

    resized_images = []
    resized_landmarks = []
    for i, image in enumerate(images):
        desired_width = int(image.width * mean_width / widths[i])
        desired_height = int(image.height * mean_height / heights[i])
        set_size = desired_width, desired_height
        resized_image = image.resize(set_size)
        # Resize landmarks along y axis:
        y_rszd_landmarks = resize_landmarks(landmarks[i], image.height, desired_height, axis=0)
        # Resize landmarks along x axis:
        new_landmarks = resize_landmarks(y_rszd_landmarks, image.width, desired_width, axis=1)
        # Save to a list
        resized_images.append(resized_image)
        resized_landmarks.append(new_landmarks)
        # Plot what you've done
        # plt.imshow(np.asarray(resized_image), cmap='gray')
        # plt.plot(new_landmarks[0:5], new_landmarks[5:10], 'r.')
        # plt.show()
    del data_set["train"]["image"], data_set["test"]["image"], \
        data_set["train"]["landmark"], data_set["test"]["landmark"]

    data_set["train"]["image"] = resized_images[0:train_size]
    data_set["test"]["image"] = resized_images[train_size:]
    data_set["train"]["landmark"] = resized_landmarks[0:train_size]
    data_set["test"]["landmark"] = resized_landmarks[train_size:]

    return data_set


def zero_slope(data_set):
    """
        Enforce zero slope over the images.
    """
    images = data_set["train"]["image"] + data_set["test"]["image"]
    landmarks = data_set["train"]["landmark"] + data_set["test"]["landmark"]
    train_size = len(data_set["train"]["image"])

    rotated_images = []
    rotated_landmarks = []
    for i, image in enumerate(images):
        current_slope = (landmarks[i][6]-landmarks[i][5]) / (landmarks[i][1] - landmarks[i][0])
        theta = current_slope * (-1)
        rotated_image = image.rotate(theta)
        rotated_landmark = rotate_landmarks(landmarks[i], theta)
        # Save to the list
        rotated_images.append(rotated_image)
        rotated_landmarks.append(rotated_landmark)
        # plt.imshow(np.asarray(rotated_image), cmap='gray')
        # plt.plot(rotated_landmark[0:5], rotated_landmark[5:10], 'r.')
        # plt.show()

    del data_set["train"]["image"], data_set["test"]["image"], \
        data_set["train"]["landmark"], data_set["test"]["landmark"]

    data_set["train"]["image"] = rotated_images[0:train_size]
    data_set["test"]["image"] = rotated_images[train_size:]
    data_set["train"]["landmark"] = rotated_landmarks[0:train_size]
    data_set["test"]["landmark"] = rotated_landmarks[train_size:]

    return data_set


def face_allignment(data_set):
    """
        Allign faces to overlap.
        :param data_set:
        :return:
    """
    images = data_set["train"]["image"] + data_set["test"]["image"]
    landmarks = data_set["train"]["landmark"] + data_set["test"]["landmark"]
    train_size = len(data_set["train"]["image"])
    widening_size = 100

    alligned_images = []
    alligned_landmarks = []

    for i, image in enumerate(images):
        crop_width = landmarks[i][1] - landmarks[i][0]
        crop_height = landmarks[i][8] - landmarks[i][5]
        bbox = (landmarks[i][0]-widening_size/2, landmarks[i][5]-widening_size/2,
                landmarks[i][0] + crop_width + widening_size/2,
                landmarks[i][5] + crop_height + widening_size/2)
        aligned_image = image.crop(bbox)
        alligned_images.append(aligned_image)
        # Align landmarks:
        origin_x = landmarks[i][0]
        origin_y = landmarks[i][5]
        xs = [x_lndmrk-origin_x + widening_size/2 for x_lndmrk in landmarks[i][0:5]]
        ys = [y_lndmrk-origin_y + widening_size/2 for y_lndmrk in landmarks[i][5:10]]
        alligned_landmarks.append(xs+ys)
        # plt.imshow(np.asarray(aligned_image), cmap='gray')
        # plt.plot(xs, ys, 'r.')
        # plt.show()

    del data_set["train"]["image"], data_set["test"]["image"], \
        data_set["train"]["landmark"], data_set["test"]["landmark"]

    data_set["train"]["image"] = alligned_images[0:train_size]
    data_set["train"]["landmark"] = alligned_landmarks[0:train_size]
    data_set["test"]["image"] = alligned_images[train_size:]
    data_set["test"]["landmark"] = alligned_landmarks[train_size:]

    return data_set


def resize_to_std(data_set, image_size):
    """
        Resize all the images to a specified size.
        :param data_set:
        :return:
    """
    images = data_set["train"]["image"] + data_set["test"]["image"]
    landmarks = data_set["train"]["landmark"] + data_set["test"]["landmark"]
    train_size = len(data_set["train"]["image"])
    resized_images = []
    resized_landmarks = []

    for i, image in enumerate(images):
        set_size = image_size, image_size
        final_image = image.resize(set_size)
        # Resize landmarks along y axis:
        y_rszd_landmarks = resize_landmarks(landmarks[i], image.height, image_size, axis=0)
        # Resize landmarks along x axis:
        new_landmarks = resize_landmarks(y_rszd_landmarks, image.width, image_size, axis=1)
        resized_images.append(final_image)
        resized_landmarks.append(new_landmarks)
        # plt.imshow(np.asarray(final_image), cmap='gray')
        # plt.plot(new_landmarks[0:5], new_landmarks[5:10], 'g.')
        # plt.show()
    del data_set["train"]["image"], data_set["test"]["image"], \
        data_set["train"]["landmark"], data_set["test"]["landmark"]

    data_set["train"]["image"] = resized_images[0:train_size]
    data_set["test"]["image"] = resized_images[train_size:]
    data_set["train"]["landmark"] = resized_landmarks[0:train_size]
    data_set["test"]["landmark"] = resized_landmarks[train_size:]

    return data_set


def gcn(image):
    """
        Global contrast normalization for images.
    :param image:
    :return:
    """
    # visit https://datascience.stackexchange.com/questions/15110/how-to-implement-global-contrast-normalization-in-python/17010
    # import numpy
    # import scipy
    # import scipy.misc
    # from PIL import Image
    # X = numpy.array(Image.open(filename))
    # replacement for the loop
    # X_average = numpy.mean(X)
    # print('Mean: ', X_average)
    # X = X - X_average
    # # `su` is here the mean, instead of the sum
    # contrast = numpy.sqrt(lmda + numpy.mean(X ** 2))
    # X = s * X / max(contrast, epsilon)
    # scipy can handle it
    # scipy.misc.imsave('result.jpg', X)


def standartize(data_set):
    """
        For standarization we use z-score(standart score normalization)
    :param data_set:
    :return:
    """
    images = data_set["train"]["image"] + data_set["test"]["image"]
    landmarks = data_set["train"]["landmark"] + data_set["test"]["landmark"]
    train_size = len(data_set["train"]["image"])

    std_images = []
    std_landmarks = []

    for image in images:
        gray_image = tf.image.rgb_to_grayscale(image)
        std_image = tf.image.per_image_standardization(gray_image)
        std_images.append(std_image)

    for landmark in landmarks:
        std_landmarks.append(normalize_landmarks(landmark))
    del data_set["train"]["image"], data_set["test"]["image"], \
        data_set["train"]["landmark"], data_set["test"]["landmark"]

    data_set["train"]["image"] = std_images[0:train_size]
    data_set["test"]["image"] = std_images[train_size:]
    data_set["train"]["landmark"] = std_landmarks[0:train_size]
    data_set["test"]["landmark"] = std_landmarks[train_size:]

    return data_set


def read_images_and_combine_others(dataset, image_size, log=False):
    """
    :param dataset:
    :param as_numpy:
    :return:
    """
    if log:
        # warn me
        print("You are logging read and combine py please be aware")
        # Create and check log file
        log_file = os.path.join(FLAGS.log_dir, "read_images_and_combine_logs")
        if os.path.exists(log_file):
            # Delete previous
            rmtree(log_file)
        if not os.path.exists(log_file):
            os.makedirs(log_file)

    train_image_path = dataset["train"]["file_path"]
    train_landmarks = dataset["train"]["landmarks"]
    train_poses = dataset["train"]["poses"]

    test_image_path = dataset["test"]["file_path"]
    test_landmarks = dataset["test"]["landmarks"]
    test_poses = dataset["test"]["poses"]

    train_size = len(train_image_path)

    f_data_set = {"file_path": [], "image": [], "landmark": [], "pose": []}

    image_paths = train_image_path + test_image_path
    landmarks = train_landmarks + test_landmarks
    poses = train_poses + test_poses

    for i, image_path in enumerate(image_paths):
        f_data_set["file_path"].append(image_path)
        image = Image.open(image_path)
        # Convert to gray scale
        image = image.convert('L')
        # Resize image: mandatory
        set_size = image_size, image_size
        image_resized = image.resize(set_size)
        # Resize landmarks along y axis:
        y_rszd_landmarks = resize_landmarks(landmarks[i], image.height, image_size, axis=0)
        # Resize landmarks along x axis:
        new_landmarks = resize_landmarks(y_rszd_landmarks, image.width, image_size, axis=1)
        if log:
            plt.imshow(np.asarray(image_resized), cmap='gray', interpolation="bicubic")
            plt.plot(new_landmarks[0:5], new_landmarks[5:10], 'g.')
            plt.savefig(os.path.join(log_file, str(i+1)+".jpg"))
            plt.close()
        f_data_set["image"].append(image_resized)
        f_data_set["landmark"].append(new_landmarks)
        f_data_set["pose"].append(poses[i])
        # Print sizes
        # print(' x '.join(str(dim) for dim in image.size))
        image.close()

    # Train test split
    train_data = {"file_path": f_data_set["file_path"][0:train_size],
                  "image": f_data_set["image"][0:train_size],
                  "landmark": f_data_set["landmark"][0:train_size],
                  "pose": f_data_set["pose"][0:train_size]}
    test_data = {"file_path": f_data_set["file_path"][train_size:],
                 "image": f_data_set["image"][train_size:],
                 "landmark": f_data_set["landmark"][train_size:],
                 "pose": f_data_set["pose"][train_size:]}
    return {"train": train_data, "test": test_data}
