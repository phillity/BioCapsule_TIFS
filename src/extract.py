import os
import cv2
from time import time
import h5py
import numpy as np
from argparse import ArgumentParser
from model.face_model import FaceNet, ArcFace


class ImageGenerator:
    def __init__(self, dataset_path, batch_size):
        self.__image_files = self.__file_walk(dataset_path)
        self.batch_size = batch_size
        self.__batches = [self.__image_files[i:i + self.batch_size]
                          for i in range(0, len(self.__image_files), self.batch_size)]
        self.image_cnt = len(self.__image_files)
        self.batch_cnt = len(self.__batches)

    def __file_walk(self, path):
        dirs, files = [], []
        contents = os.listdir(path)
        for content in contents:
            if os.path.isfile(os.path.join(path, content)):
                files += [(os.path.join(path, content),
                           os.path.basename(path))]
            else:
                dirs += [content]
        for d in dirs:
            files += self.__file_walk(os.path.join(path, d))
        return files

    def get_batch(self, idx):
        batch = self.__batches[idx]
        x = np.array([self.__resize(cv2.imread(xy[0])) for xy in batch])
        y = np.array([xy[1] for xy in batch])
        return (x, y)

    def __resize(self, image, image_size=320, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # image height and width are the same
        if h == w:
            return cv2.resize(image, (image_size, image_size), interpolation=inter)

        # if image height is greater than height
        elif h > w:
            r = image_size / float(h)
            dim = (int(w * r), image_size)
            resized = cv2.resize(image, dim, interpolation=inter)
            if (image_size - resized.shape[1]) % 2 == 0:
                pad = int((image_size - resized.shape[1]) / 2)
                return cv2.copyMakeBorder(resized, 0, 0, pad, pad, cv2.BORDER_CONSTANT, 0)
            else:
                pad = int((image_size - resized.shape[1]) / 2)
                return cv2.copyMakeBorder(resized, 0, 0, pad, pad + 1, cv2.BORDER_CONSTANT, 0)

        # if image width is greater than height
        else:
            r = image_size / float(w)
            dim = (image_size, int(h * r))
            resized = cv2.resize(image, dim, interpolation=inter)
            if (image_size - resized.shape[0]) % 2 == 0:
                pad = int((image_size - resized.shape[0]) / 2)
                return cv2.copyMakeBorder(resized, pad, pad, 0, 0,  cv2.BORDER_CONSTANT, 0)
            else:
                pad = int((image_size - resized.shape[0]) / 2)
                return cv2.copyMakeBorder(resized, pad, pad + 1, 0, 0,  cv2.BORDER_CONSTANT, 0)


def progress_bar(text, percent, barLen=20):
    print("{} -- [{:<{}}] {:.0f}%".format(
        text, "=" * int(barLen * percent), barLen, percent * 100), end="\r"
    )
    if percent == 1:
        print("\n")


def extract(dataset, method, gpu, batch_size):
    if method == "arcface":
        model = ArcFace(gpu)
    if method == "facenet":
        model = FaceNet(gpu)

    image_gen = ImageGenerator(os.path.join(
        os.path.abspath(""), "image", dataset), batch_size)

    fi_out = h5py.File(os.path.join(os.path.abspath(
        ""), "data", method, "{}.hdf5".format(dataset)), "w")
    X = fi_out.create_dataset(
        "X", (image_gen.image_cnt, 512), dtype="f", compression="lzf")
    y = fi_out.create_dataset(
        "y", (image_gen.image_cnt,), dtype=h5py.string_dtype(encoding="ascii"), compression="lzf")

    cnt = 0
    start = time()
    for i in range(image_gen.batch_cnt):
        images, labels = image_gen.get_batch(i)
        features = model.embed(images)

        for j in range(features.shape[0]):
            X[cnt] = features[j]
            y[cnt] = labels[j]
            cnt += 1

        end = time()
        avg_time = (end - start) / cnt
        progress_bar("Extracting {} using {} -- Batch {}/{} -- Time per image {:03f}sec".format(
            dataset, method, (i + 1), image_gen.batch_cnt, avg_time / image_gen.batch_size), (i + 1) / image_gen.batch_cnt)

    fi_out.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True,
                        help="dataset to use in feature extraction")
    parser.add_argument("-m", "--method", required=True, choices=["arcface", "facenet"],
                        help="method to use in feature extraction")
    parser.add_argument("-gpu", "--gpu", required=False, type=int, default=-1,
                        help="gpu to use in feature extraction")
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=32,
                        help="batch size to use in feature extraction")
    args = vars(parser.parse_args())
    extract(args["dataset"], args["method"], args["gpu"], args["batch_size"])
