import os
import h5py
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from biocapsule import BioCapsuleGenerator


np.random.seed(42)
tf.compat.v1.set_random_seed(42)


def get_clf(input_shape, output_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(output_shape)(inputs)
    predictions = Activation("softmax")(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["acc"])
    return model


def train_clf(clf, X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    class_weights = compute_class_weight(
        "balanced", np.unique(y_train), y_train)
    es = EarlyStopping(monitor="val_acc", mode="max",
                       verbose=0, patience=1, restore_best_weights=True)
    clf.fit(x=X_train, y=y_train, epochs=100000, batch_size=128,
            verbose=2, callbacks=[es], validation_data=[X_val, y_val],
            class_weight=class_weights)
    return clf


def identification(dataset, method, representation):
    # load data and setup output file
    out_file = open(os.path.join(os.path.abspath(""), "results",
                                 "identification_{}_{}_{}.txt".format(dataset, method, representation)), "w")
    feature_dataset = h5py.File(os.path.join(os.path.abspath(
        ""), "data", method, "{}.hdf5".format(dataset)), "r")
    rs_dataset = h5py.File(os.path.join(os.path.abspath(
        ""), "data", method, "rs.hdf5"), "r")
    X, y = feature_dataset["X"][:], feature_dataset["y"][:]
    y = LabelEncoder().fit_transform(np.array([label.decode() for label in y]))
    rs_feature = rs_dataset["X"][0]

    # remove subjects from lfw dataset with less than 5 images
    if dataset == "lfw":
        y_uni, y_cnt = np.unique(y, return_counts=True)
        mask = np.array(
            [idx for idx, label in enumerate(y) if label not in y_uni[y_cnt < 5]])
        X, y = X[mask], y[mask]
        for label_new, label in enumerate(np.unique(y)):
            y[y == label] = label_new

    # fuse features with rs to form biocapsules
    if representation == "biocapsule":
        bc_gen = BioCapsuleGenerator()
        X_train = bc_gen.biocapsule_batch(X, rs_feature)

    # 5-fold cross-validation experiment
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    err, acc, pre, rec, f1s = [[] for i in range(5)]
    for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # train model, if using vggface2 dataset use tf model, otherwise use sklearn model
        print("{} {} {} -- identification -- fold {} -- training".format(dataset,
                                                                         method, representation, k))
        X_train, y_train = X[train_idx], y[train_idx]
        if dataset == "vggface2":
            clf = get_clf(512, np.unique(y_train).shape[0])
            clf = train_clf(clf, X_train, y_train)
        else:
            clf = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42)
            cv_results = cross_validate(
                clf, X_train, y_train, cv=3, return_estimator=True)
            clf = cv_results["estimator"][np.argmax(cv_results["test_score"])]
        del X_train, y_train

        # evaluate model, if using vggface2 dataset use tf model, otherwise use sklearn model
        print("{} {} {} -- identification -- fold {} -- testing".format(dataset,
                                                                        method, representation, k))
        X_test, y_test = X[test_idx], y[test_idx]
        if dataset == "vggface2":
            y_pred = np.argmax(clf.predict(X_test), axis=-1)
        else:
            y_pred = clf.predict(X_test)
        del X_test
        
        # accumulate results
        err.append(np.sum(y_test != y_pred))
        acc.append(accuracy_score(y_test, y_pred))
        pre.append(precision_score(y_test, y_pred, average="macro"))
        rec.append(recall_score(y_test, y_pred, average="macro"))
        f1s.append(f1_score(y_test, y_pred, average="macro"))

        # write cross-validation fold results
        out_file.write("Fold {}:\n".format(k))
        out_file.write("ERR -- {}\n".format(err[-1]))
        out_file.write("ACC -- {:.6f}\n".format(100. * acc[-1]))
        out_file.write("PRE -- {:.6f}\n".format(100. * pre[-1]))
        out_file.write("REC -- {:.6f}\n".format(100. * rec[-1]))
        out_file.write("F1  -- {:.6f}\n".format(f1s[-1]))
        out_file.flush()

    # write overall 5-fold cross-validation results
    out_file.write("Overall:\n")
    out_file.write(
        "ERR -- {}\n".format(np.sum(err)))
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
    for dataset in ["yalefaces", "yalefacesb", "gtdb", "caltech", "fei", "feret_color", "cmu_p", "cmu_i", "cmu_e", "cmu_l", "lfw", "vggface2"]:
        for method in ["facenet", "arcface"]:
            for representation in ["feature", "biocapsule"]:
                identification(dataset, method, representation)
