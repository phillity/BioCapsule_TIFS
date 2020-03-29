import os
import cv2
from time import time
import h5py
import numpy as np
from argparse import ArgumentParser
from model.face_model import FaceNet, ArcFace


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


def extract(dataset, method, gpu):
    if method == "arcface":
        model = ArcFace(gpu)
    if method == "facenet":
        model = FaceNet(gpu)

    cnt = 0
    image_cnt = file_walk(os.path.join(os.path.abspath(""), "image", dataset))

    fi_out = h5py.File(os.path.join(os.path.abspath(
        ""), "data", method, "{}.hdf5".format(dataset)), "w")
    X = fi_out.create_dataset(
        "X", (image_cnt, 512), dtype="f", compression="lzf")
    y = fi_out.create_dataset(
        "y", (image_cnt,), dtype="i", compression="lzf")

    start = time()
    
    subjects = os.listdir(os.path.join(os.path.abspath(""), "image", dataset))
    subjects = [x for _, x in sorted(
        zip([subject.lower() for subject in subjects], subjects))]
    for subject_y, subject_dir in enumerate(subjects):
        for image_fi in os.listdir(os.path.join(os.path.abspath(""), "image", dataset, subject_dir)):
            image = cv2.imread(os.path.join(
                os.path.abspath(""), "image", dataset, subject_dir, image_fi))

            X[cnt] = model.embed(image)
            y[cnt] = subject_y + 1

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

    extract(args["dataset"], args["method"], args["gpu"])
