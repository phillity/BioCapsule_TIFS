from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import mxnet as mx
import tensorflow as tf
from sklearn.preprocessing import normalize
from model import facenet
from model.face_preprocess import preprocess
from model.mtcnn_detector import MtcnnDetector


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1.0 / std_adj)
    return y


class FaceNet:
    def __init__(self, gpu):
        if gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu)

        self.__sess = tf.compat.v1.Session()
        facenet.load_model(os.path.join(os.path.abspath(
            ""), "src", "model", "facenet", "20180402-114759.pb"))
        self.__images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "input:0")
        self.__embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "embeddings:0")
        self.__phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "phase_train:0")
        self.__detector = MtcnnDetector(model_folder=os.path.join(
            os.path.abspath(""), "src", "model", "mtcnn"), ctx=ctx, accurate_landmark=True)

    def embed(self, images):
        align = np.zeros((images.shape[0], 160, 160, 3))
        for i, image in enumerate(images):
            dets = self.__detector.detect_face(image)

            # if no faces detected in image, get embedding for entire image
            if dets is None:
                image = cv2.resize(image, (112, 112))

            # if multiple faces detected in image, get embedding for centermost face
            elif dets[0].shape[0] > 1:
                bboxs, landmarks = dets
                image_center = np.array([image.shape[0], image.shape[1]]) / 2
                det_centers = [np.array([(bbox[1] + bbox[3]),
                                        (bbox[0] + bbox[2])]) / 2 for bbox in bboxs]
                dists = [np.linalg.norm(image_center - det_center)
                        for det_center in det_centers]

                landmark = landmarks[np.argmin(dists)].reshape((2, 5)).T
                bbox = bboxs[np.argmin(dists)][:4]
                image = preprocess(image, bbox, landmark, image_size="112,112")

            else:
                landmark = dets[1].reshape((2, 5)).T
                bbox = dets[0][0, :4]
                image = preprocess(image, bbox, landmark, image_size="112,112")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            align[i] = prewhiten(cv2.resize(image, (160, 160)))

        feed_dict = {self.__images_placeholder: align,
                     self.__phase_train_placeholder: False}
        embedding = self.__sess.run(
            self.__embeddings, feed_dict=feed_dict)
        embedding = normalize(embedding)
        return embedding


class ArcFace:
    def __init__(self, gpu):
        if gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu)
        self.__model = self.__get_model(ctx)
        self.__detector = MtcnnDetector(model_folder=os.path.join(
            os.path.abspath(""), "src", "model", "mtcnn"), ctx=ctx, accurate_landmark=True)

    def __get_model(self, ctx):
        model_path = os.path.join(
            os.path.abspath(""), "src", "model", "arcface", "model")
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
        all_layers = sym.get_internals()
        sym = all_layers["fc1_output"]
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[("data", (1, 3, 112, 112))])
        model.set_params(arg_params, aux_params)
        return model

    def embed(self, images):
        align = np.zeros((images.shape[0], 3, 112, 112))
        for i, image in enumerate(images):
            dets = self.__detector.detect_face(image)

            # if no faces detected in image, get embedding for entire image
            if dets is None:
                image = cv2.resize(image, (112, 112))

            # if multiple faces detected in image, get embedding for centermost face
            elif dets[0].shape[0] > 1:
                bboxs, landmarks = dets
                image_center = np.array([image.shape[0], image.shape[1]]) / 2
                det_centers = [np.array([(bbox[1] + bbox[3]),
                                        (bbox[0] + bbox[2])]) / 2 for bbox in bboxs]
                dists = [np.linalg.norm(image_center - det_center)
                        for det_center in det_centers]

                landmark = landmarks[np.argmin(dists)].reshape((2, 5)).T
                bbox = bboxs[np.argmin(dists)][:4]
                image = preprocess(image, bbox, landmark, image_size="112,112")

            else:
                landmark = dets[1].reshape((2, 5)).T
                bbox = dets[0][0, :4]
                image = preprocess(image, bbox, landmark, image_size="112,112")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            align[i] = np.transpose(image, (2, 0, 1))

        data = mx.nd.array(align)
        db = mx.io.DataBatch(data=(data,))
        self.__model.forward(db, is_train=False)
        embedding = self.__model.get_outputs()[0].asnumpy()
        embedding = normalize(embedding)
        return embedding
