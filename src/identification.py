import os
import h5py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from biocapsule import BioCapsuleGenerator


np.random.seed(42)


def identification(dataset, method, representation):
    out_file = open(os.path.join(os.path.abspath(""), "results",
                                 "identification_{}_{}_{}.txt".format(dataset, method, representation)), "w")
    feature_dataset = h5py.File(os.path.join(os.path.abspath(
        ""), "data", method, "{}.hdf5".format(dataset)), "r")
    rs_dataset = h5py.File(os.path.join(os.path.abspath(
        ""), "data", method, "rs.hdf5"), "r")

    X, y = feature_dataset["X"][:], feature_dataset["y"][:].astype(int) - 1
    rs_feature = rs_dataset["X"][0]

    if dataset == "lfw":
        y_uni, y_cnt = np.unique(y, return_counts=True)
        mask = np.array(
            [idx for idx, label in enumerate(y) if label not in y_uni[y_cnt < 5]])
        X, y = X[mask], y[mask]
        for label_new, label in enumerate(np.unique(y)):
            y[y == label] = label_new

    if representation == "biocapsule":
        bc_gen = BioCapsuleGenerator()
        X_train = bc_gen.biocapsule_batch(X, rs_feature)

    acc, pre, rec, f1s = [[] for i in range(4)]
    skf = StratifiedKFold(n_splits=5)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42)
        cv_results = cross_validate(
            clf, X_train, y_train, cv=3, return_estimator=True)
        clf = cv_results["estimator"][np.argmax(cv_results["test_score"])]
        y_pred = clf.predict(X_test)

        acc.append(accuracy_score(y_test, y_pred))
        pre.append(precision_score(y_test, y_pred, average="macro"))
        rec.append(recall_score(y_test, y_pred, average="macro"))
        f1s.append(f1_score(y_test, y_pred, average="macro"))

    out_file.write(
        "ACC -- {:.6f} (+/-{:.6f})\n".format(100. * np.average(acc), 100. * np.std(acc)))
    out_file.write(
        "PRE -- {:.6f} (+/-{:.6f})\n".format(100. * np.average(pre), 100. * np.std(pre)))
    out_file.write(
        "REC -- {:.6f} (+/-{:.6f})\n".format(100. * np.average(rec), 100. * np.std(rec)))
    out_file.write(
        "F1  -- {:.6f} (+/-{:.6f})\n".format(np.average(f1s), np.std(f1s)))
    out_file.close()


if __name__ == "__main__":
    for dataset in ["gtdb", "caltech", "lfw"]:  # vggface2
        for method in ["facenet", "arcface"]:
            for representation in ["feature", "biocapsule"]:
                identification(dataset, method, representation)
