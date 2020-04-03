import os
import csv
import h5py
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from biocapsule import BioCapsuleGenerator


np.random.seed(42)


def get_lfw(method, representation):
    people = []
    with open(os.path.join(os.path.abspath(""), "image", "people.txt"), "r") as people_file:
        people_list = list(csv.reader(people_file, delimiter="\t"))
        assert(len(people_list[2:603]) == 601)
        people.append(people_list[2:603])
        assert(len(people_list[604:1159]) == 555)
        people.append(people_list[604:1159])
        assert(len(people_list[1160:1712]) == 552)
        people.append(people_list[1160:1712])
        assert(len(people_list[1713:2273]) == 560)
        people.append(people_list[1713:2273])
        assert(len(people_list[2274:2841]) == 567)
        people.append(people_list[2274:2841])
        assert(len(people_list[2842:3369]) == 527)
        people.append(people_list[2842:3369])
        assert(len(people_list[3370:3967]) == 597)
        people.append(people_list[3370:3967])
        assert(len(people_list[3968:4569]) == 601)
        people.append(people_list[3968:4569])
        assert(len(people_list[4570:5150]) == 580)
        people.append(people_list[4570:5150])
        assert(len(people_list[5151:]) == 609)
        people.append(people_list[5151:])

    pairs = []
    with open(os.path.join(os.path.abspath(""), "image", "pairs.txt"), "r") as pairs_file:
        pairs_list = list(csv.reader(pairs_file, delimiter="\t"))
        for i in range(10):
            idx = i * 600 + 1
            pairs.append(pairs_list[idx: idx + 600])
            assert (len(pairs[i]) == 600)

    feature_dataset = h5py.File(os.path.join(os.path.abspath(
        ""), "data", method, "lfw.hdf5"), "r")
    rs_dataset = h5py.File(os.path.join(os.path.abspath(
        ""), "data", method, "rs.hdf5"), "r")

    X, y = feature_dataset["X"][:], feature_dataset["y"][:]
    y = np.array([label.decode() for label in y])
    rs_feature = rs_dataset["X"][0]

    if representation == "biocapsule":
        bc_gen = BioCapsuleGenerator()
        X_train = bc_gen.biocapsule_batch(X, rs_feature)

    subject = {}
    for s_id, s in enumerate(os.listdir(os.path.join(
            os.path.abspath(""), "image", "lfw"))):
        subject[s] = s_id

    lfw = {}
    for i in range(10):
        train = people[i]
        train_cnt = np.sum([int(s[-1]) for s in train])
        test = pairs[i]

        lfw["train_{}".format(i)] = np.zeros((train_cnt, 513))
        lfw["test_{}".format(i)] = np.zeros((600, 2, 513))

        train_idx = 0
        for s in train:
            s_id = subject[s[0]]
            s_features = X[y == s[0]]
            assert (s_features.shape[0] == int(s[1]))

            for j in range(s_features.shape[0]):
                lfw["train_{}".format(i)][train_idx] = np.append(s_features[j], s_id)
                train_idx += 1

        assert (train_idx == train_cnt)

        for test_idx, s in enumerate(test):
            if len(s) == 3:
                s_id = subject[s[0]]
                s_features = X[y == s[0]]
                lfw["test_{}".format(i)][test_idx,
                                         0] = np.append(s_features[int(s[1]) - 1], s_id)
                lfw["test_{}".format(i)][test_idx,
                                         1] = np.append(s_features[int(s[2]) - 1], s_id)
            else:
                s_id_1 = subject[s[0]]
                s_features = X[y == s[0]]
                lfw["test_{}".format(i)][test_idx,
                                         0] = np.append(s_features[int(s[1]) - 1], s_id_1)
                s_id_2 = subject[s[2]]
                s_features = X[y == s[2]]
                lfw["test_{}".format(i)][test_idx,
                                         1] = np.append(s_features[int(s[3]) - 1], s_id_2)

        assert (test_idx == 599)

    return lfw


def verification(method, representation):
    out_file = open(os.path.join(os.path.abspath(""), "results",
                                 "verification_lfw_{}_{}.txt".format(method, representation)), "w")
    lfw = get_lfw(method, representation)

    fps, fns, acc, pre, rec, f1s, fpr, frr, eer, aucs = [[] for _ in range(10)]
    for fold in range(10):
        train_fold = lfw["train_{}".format(fold)]
        X_train = train_fold[:, :-1]
        y_train = train_fold[:, -1][..., np.newaxis]

        p_dist = euclidean_distances(X_train, X_train)
        p_dist = p_dist[np.triu_indices_from(p_dist, k=1)][..., np.newaxis]
        y_mat = np.equal(y_train, y_train.T).astype(int)
        y = y_mat[np.triu_indices_from(y_mat, k=1)]

        X_train = np.vstack([p_dist[y == 0][np.random.choice(p_dist[y == 0].shape[0], p_dist[y == 1].shape[0], replace=False)],
                             p_dist[y == 1]])
        y_train = np.hstack(
            [np.zeros((p_dist[y == 1].shape[0],)), np.ones((p_dist[y == 1].shape[0],))])

        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42)
        cv_results = cross_validate(
            clf, X_train, y_train, cv=3, return_estimator=True)
        clf = cv_results["estimator"][np.argmax(cv_results["test_score"])]

        test_fold = lfw["test_{}".format(fold)]
        p_dist = euclidean_distances(
            test_fold[:, 0, :][:, :-1], test_fold[:, 1, :][:, :-1])
        p_dist = p_dist.diagonal()[..., np.newaxis]
        y_test = np.hstack([np.ones((300,)), np.zeros((300,))])

        y_prob = clf.predict_proba(p_dist)[:, 1]
        y_pred = np.around(y_prob)
        _, fp, fn, _ = confusion_matrix(y_test, y_pred).ravel()
        fprf, tprf, thresholds = roc_curve(y_test, y_prob)

        fps.append(fp)
        fns.append(fn)
        acc.append(accuracy_score(y_test, y_pred))
        pre.append(precision_score(y_test, y_pred))
        rec.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        fpr.append(1. - sensitivity_score(y_test, y_pred))
        frr.append(1. - specificity_score(y_test, y_pred))
        eer.append(brentq(lambda x: 1. - x - interp1d(fprf, tprf)(x), 0., 1.))
        aucs.append(auc(fprf, tprf))

        out_file.write("Fold {}:\n".format(fold))
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
    for method in ["facenet", "arcface"]:
        for representation in ["feature", "biocapsule"]:
            verification(method, representation)
