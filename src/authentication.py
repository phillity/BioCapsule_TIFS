import os
import h5py
import numpy as np
from queue import Queue
from threading import Thread
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from argparse import ArgumentParser
from biocapsule import BioCapsuleGenerator


np.random.seed(42)


def train_test_clf(c, X_train, y_train, X_test, y_test, queue):
    y_train_bin = np.zeros(y_train.shape)
    y_train_bin[y_train == c] = 1
    y_test_bin = np.zeros(y_test.shape)
    y_test_bin[y_test == c] = 1

    clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42)
    cv_results = cross_validate(
        clf, X_train, y_train_bin, cv=3, return_estimator=True)
    clf = cv_results["estimator"][np.argmax(cv_results["test_score"])]
    y_prob = clf.predict_proba(X_test)[:, 1]

    queue.put((c, y_test_bin, y_prob))


def authentication(dataset, method, representation, thread_cnt):
    # load data and setup output file
    out_file = open(os.path.join(os.path.abspath(""), "results",
                                "authentication_{}_{}_{}.txt".format(dataset, method, representation)), "w")
    feature_dataset = h5py.File(os.path.join(os.path.abspath(
        ""), "data", method, "{}.hdf5".format(dataset)), "r")
    rs_dataset = h5py.File(os.path.join(os.path.abspath(
        ""), "data", method, "rs.hdf5"), "r")
    X, y = feature_dataset["X"][:], feature_dataset["y"][:]
    y = LabelEncoder().fit_transform(np.array([label.decode() for label in y])) + 1
    rs_feature = rs_dataset["X"][0]

    # remove subjects from lfw dataset with less than 5 images
    if dataset == "lfw":
        y_uni, y_cnt = np.unique(y, return_counts=True)
        mask = np.array(
            [idx for idx, label in enumerate(y) if label not in y_uni[y_cnt < 5]])
        X, y = X[mask], y[mask]
        for label_new, label in enumerate(np.unique(y)):
            y[y == label] = label_new + 1

    # fuse features with rs to form biocapsules
    if representation == "biocapsule":
        bc_gen = BioCapsuleGenerator()
        X_train = bc_gen.biocapsule_batch(X, rs_feature)

    # 5-fold cross-validation experiment
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fps, fns, acc, pre, rec, f1s, fpr, frr, eer, aucs = [[] for _ in range(10)]
    for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # setup for training
        print("{} {} {} -- authentication -- fold {}".format(dataset,
                                                             method, representation, k))
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # split classes so we can spawn threads to train each binary classifier
        classes = np.unique(y_train)
        classes_split = [classes[i:i + thread_cnt] for i in range(0, classes.shape[0], thread_cnt)]
        
        # train binary classifiers
        y_prob = np.zeros((X_test.shape[0] * classes.shape[0],))
        y_true = np.zeros((X_test.shape[0] * classes.shape[0],))
        for li in classes_split:
            threads = []
            queue = Queue()
            for c in li:
                threads.append(Thread(target=train_test_clf, args=(
                    c, X_train, y_train, X_test, y_test, queue)))
                threads[-1].start()
            _ = [t.join() for t in threads]
            while not queue.empty():
                (c_idx, true, prob) = queue.get()
                c_idx = int(c_idx - 1) * X_test.shape[0]
                y_true[c_idx: c_idx + X_test.shape[0]] = true
                y_prob[c_idx: c_idx + X_test.shape[0]] = prob
        del X_train, y_train
        del X_test, y_test

        y_pred = np.around(y_prob)
        _, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
        fprf, tprf, thresholds = roc_curve(y_true, y_prob)
        
        fps.append(fp)
        fns.append(fn)
        acc.append(accuracy_score(y_true, y_pred))
        pre.append(precision_score(y_true, y_pred))
        rec.append(recall_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred))
        fpr.append(1. - sensitivity_score(y_true, y_pred))
        frr.append(1. - specificity_score(y_true, y_pred))
        eer.append(brentq(lambda x: 1. - x - interp1d(fprf, tprf)(x), 0., 1.))
        aucs.append(auc(fprf, tprf))

        out_file.write("Fold {}:\n".format(k))
        out_file.write("FP -- {}, FN -- {}\n".format(fps[-1], fns[-1]))
        out_file.write("ACC -- {:.6f}\n".format(100. * acc[-1]))
        out_file.write("PRE -- {:.6f}\n".format(100. * pre[-1]))
        out_file.write("REC -- {:.6f}\n".format(100. * rec[-1]))
        out_file.write("F1  -- {:.6f}\n".format(f1s[-1]))
        out_file.write("FPR -- {:.6f}\n".format(100. * fpr[-1]))
        out_file.write("FRR -- {:.6f}\n".format(100. * frr[-1]))
        out_file.write("EER -- {:.6f}\n".format(100. * eer[-1]))
        out_file.write("AUC -- {:.6f}\n".format(aucs[-1]))
        out_file.flush()

    out_file.write("Overall:\n")
    out_file.write("FP -- {}, FN -- {}\n".format(np.sum(fps), np.sum(fns)))
    out_file.write(
        "ACC -- {:.6f} (+/-{:.6f})\n".format(100. * np.average(acc), 100. * np.std(acc)))
    out_file.write(
        "PRE -- {:.6f} (+/-{:.6f})\n".format(100. * np.average(pre), 100. * np.std(pre)))
    out_file.write(
        "REC -- {:.6f} (+/-{:.6f})\n".format(100. * np.average(rec), 100. * np.std(rec)))
    out_file.write(
        "F1  -- {:.6f} (+/-{:.6f})\n".format(np.average(f1s), np.std(f1s)))
    out_file.write(
        "FPR -- {:.6f} (+/-{:.6f})\n".format(100. * np.average(fpr), 100. * np.std(fpr)))
    out_file.write(
        "FRR -- {:.6f} (+/-{:.6f})\n".format(100. * np.average(frr), 100. * np.std(frr)))
    out_file.write(
        "EER -- {:.6f} (+/-{:.6f})\n".format(100. * np.average(eer), 100. * np.std(eer)))
    out_file.write(
        "AUC -- {:.6f} (+/-{:.6f})\n".format(np.average(aucs), np.std(aucs)))
    out_file.close()


if __name__ == "__main__":
    thread_cnt = 8
    for dataset in ["yalefaces", "yalefacesb", "gtdb", "caltech", "fei", "feret_color", "cmu_p", "cmu_i", "cmu_e", "cmu_l", "lfw"]:
        for method in ["facenet", "arcface"]:
            for representation in ["feature", "biocapsule"]:
                authentication(dataset, method, representation, thread_cnt)
