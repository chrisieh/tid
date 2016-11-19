#!/usr/bin/env python
import argparse
import tid.evaluation as ev

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input TMVA root-file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    wp = [0.3, 0.5, 0.7]

    truth, score, weight = ev.load_tmva(args.file)
    fpr, tpr, thr = ev.tmva_roc(args.file)

    for eff in wp:
        bdt_thr = ev.bdt_threshold(tpr, thr, eff)
        rej, err = ev.rejection(score[truth == 0], weight[truth == 0], bdt_thr)
        print("Rej. @{} eff.: {} +- {}".format(eff, rej, err))

