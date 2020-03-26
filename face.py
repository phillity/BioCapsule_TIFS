import os
import cv2
from time import time
import h5py
import numpy as np
import mxnet as mx
import tensorflow as tf
from skimage import transform as trans
from argparse import ArgumentParser
from insightface.app.face_analysis import Face, FaceAnalysis
from model.facenet import facenet
from model.mtcnn.mtcnn import MtcnnDetector


class ArcFace:
    def __init__(self, gpu_id):
        self.__model = FaceAnalysis(
            det_name="retinaface_r50_v1",
            rec_name="arcface_r100_v1",
            ga_name=None
        )
        self.__model.prepare(gpu_id)

    def __retinaface_align(self, image, landmark):
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
        src = np.expand_dims(src, axis=0)

        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(landmark, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float("inf")

        for i in np.arange(src.shape[0]):
            tform.estimate(landmark, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i

        warped = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
        return warped

    def embed(self, image):
        dets = self.__model.det_model.detect(
            image, threshold=0.8, scale=1.0)

        # if no faces detected in image, get embedding for entire image
        if len(dets[0]) == 0:
            image = cv2.resize(image, (112, 112))

        # if multiple faces detected in image, get embedding for centermost face
        elif len(dets[0]) > 1:
            bboxs, landmarks = dets
            image_center = np.array([image.shape[0], image.shape[1]]) / 2
            det_centers = [np.array([(bbox[3] + bbox[1]),
                                     (bbox[2] + bbox[0])]) / 2 for bbox in bboxs]
            dists = [np.linalg.norm(image_center - det_center)
                     for det_center in det_centers]

            landmark = landmarks[np.argmin(dists)]
            image = self.__retinaface_align(image, landmark)

        # if only one face detected in image, get its embedding
        else:
            landmark = dets[1][0]
            image = self.__retinaface_align(image, landmark)

        embedding = self.__model.rec_model.get_embedding(image).flatten()
        return embedding / np.linalg.norm(embedding)


class FaceNet:
    def __init__(self, gpu):
        if gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu)

        self.__sess = tf.compat.v1.Session()
        model_path = os.path.join(
            os.path.abspath(""), "model", "facenet", "20180408-102900.pb")
        facenet.load_model(model_path)
        self.__images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "input:0")
        self.__embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "embeddings:0")
        self.__phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "phase_train:0")

        self.__detector = MtcnnDetector(model_folder=os.path.join(
            os.path.abspath(""), "model", "mtcnn"), ctx=ctx, accurate_landmark=True)

    def __mtcnn_align(self, image, landmark):
        M = None
        src = np.array([[30.2946, 51.6963],
                        [65.5318, 51.5014],
                        [48.0252, 71.7366],
                        [33.5493, 92.3655],
                        [62.7299, 92.2041]],
                       dtype=np.float32)
        src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]

        warped = cv2.warpAffine(
            image, M, (112, 112), borderValue=0.0)
        return warped

    def __prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1.0 / std_adj)
        return y

    def embed(self, image):
        dets = self.__detector.detect_face(image)

        # if no faces detected in image, get embedding for entire image
        if dets is None:
            pass

        # if multiple faces detected in image, get embedding for centermost face
        elif dets[0].shape[0] > 1:
            bboxs, landmarks = dets
            image_center = np.array([image.shape[0], image.shape[1]]) / 2
            det_centers = [np.array([(bbox[1] + bbox[3]),
                                     (bbox[0] + bbox[2])]) / 2 for bbox in bboxs]
            dists = [np.linalg.norm(image_center - det_center)
                     for det_center in det_centers]

            landmark = landmarks[np.argmin(dists)].reshape((2, 5)).T
            image = self.__mtcnn_align(image, landmark)

        # if only one face detected in image, get its embedding
        else:
            landmark = dets[1].reshape((2, 5)).T
            image = self.__mtcnn_align(image, landmark)

        image = cv2.resize(image, (160, 160))
        input_blob = self.__prewhiten(image)
        input_blob = np.expand_dims(input_blob, axis=0)
        feed_dict = {self.__images_placeholder: input_blob,
                        self.__phase_train_placeholder: False}
        embedding = self.__sess.run(
            self.__embeddings, feed_dict=feed_dict)[0]
        return embedding / np.linalg.norm(embedding)


def progress_bar(text, percent, barLen=20):
    print("{} -- [{:<{}}] {:.0f}%".format(
        text, "=" * int(barLen * percent), barLen, percent * 100), end="\r"
    )
    if percent == 1:
        print("\n")


def file_walk(path):
    files = 0
    dirs = []
    contents = os.listdir(path)
    for content in contents:
        if os.path.isfile(os.path.join(path, content)):
            files += 1
        else:
            dirs += [content]
    for d in dirs:
        files += file_walk(os.path.join(path, d))
    return files


def extract_dataset(dataset, method, gpu):
    if method == "arcface":
        model = ArcFace(gpu)
    if method == "facenet":
        model = FaceNet(gpu)

    cnt = 0
    image_cnt = file_walk(os.path.join(os.path.abspath(""), "image", dataset))

    fi_out = h5py.File(os.path.join(os.path.abspath(
        ""), "data", method, "{}.hdf5".format(dataset)), "w")
    X = fi_out.create_dataset(
        "X", (image_cnt, 512), dtype="f", compression="gzip", compression_opts=9)
    y = fi_out.create_dataset(
        "y", (image_cnt,), dtype="i", compression="gzip", compression_opts=9)

    X_flip = fi_out.create_dataset(
        "X_flip", (image_cnt, 512), dtype="f", compression="gzip", compression_opts=9)
    y_flip = fi_out.create_dataset(
        "y_flip", (image_cnt,), dtype="i", compression="gzip", compression_opts=9)

    start = time()

    for subject_y, subject_dir in enumerate(os.listdir(os.path.join(os.path.abspath(""), "image", dataset))):
        for image_fi in os.listdir(os.path.join(os.path.abspath(""), "image", dataset, subject_dir)):
            image = cv2.imread(os.path.join(
                os.path.abspath(""), "image", dataset, subject_dir, image_fi))
            image_flip = cv2.flip(image, 1)

            X[cnt] = model.embed(image)
            y[cnt] = subject_y + 1
            X_flip[cnt] = model.embed(image_flip)
            y_flip[cnt] = subject_y + 1

            end = time()

            cnt += 1
            avg_time = (end - start) / cnt
            progress_bar("Extracting {} using {} -- Time per image {:03f}sec".format(
                dataset, method, avg_time), cnt / image_cnt)

    fi_out.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True,
                        help="dataset to use in feature extraction")
    parser.add_argument("-m", "--method", required=True, choices=["arcface", "facenet"],
                        help="method to use in feature extraction")
    parser.add_argument("-gpu", "--gpu", required=False, type=int, default=-1,
                        help="gpu to use in feature extraction")
    args = vars(parser.parse_args())

    extract_dataset(args["dataset"], args["method"], args["gpu"])
