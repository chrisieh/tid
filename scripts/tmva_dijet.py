#!/usr/bin/env python
import argparse
import tid.evaluation as ev
from root_numpy import root2array

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tmva", help="TMVA-BDT root-file")
    parser.add_argument("dijet", help="Dijet decorated root-file")
    parser.add_argument("-n", help="BDT name")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    wp = [0.3, 0.5, 0.7]

    truth, score, weight = ev.load_tmva(args.tmva)
    fpr, tpr, thr = ev.tmva_roc(args.tmva)

    data = root2array(args.dijet, treename="CollectionTree",
                      branches=[args.n, "weight"])
    score_data, score_weight = data[args.n], data["weight"]

    for eff in wp:
        bdt_thr = ev.bdt_threshold(tpr, thr, eff)
        rej, err = ev.rejection(score[truth == 0], weight[truth == 0], bdt_thr)
        rej_data, err_data = ev.rejection(score_data, score_weight, bdt_thr)
        print("[MC]: Rej. @{} eff.: {} +- {}".format(eff, rej, err))
        print("[Data]: Rej. @{} eff.: {} +- {}".format(eff, rej_data, err_data))