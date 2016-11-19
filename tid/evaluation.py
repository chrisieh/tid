import numpy as np
import root_numpy as rnp
from sklearn.metrics import roc_curve


def tmva_roc(rootf):
    """Returns fpr, tpr, thr for a TMVA-style root-file"""
    data = rnp.root2array(rootf, treename="TestTree",
                          branches=["classifier", "classID", "weight"])
    
    # Convert tmva scheme (sig 0, bkg 1) to (sig 1, bkg 0)
    truth = data["classID"]
    sig = (truth == 0)
    bkg = np.logical_not(sig)
    truth[sig] = 1
    truth[bkg] = 0

    return roc_curve(truth, data["classifier"], sample_weight=data["weight"])


def root_roc(sigf, bkgf, branch, weight="weight", tree="CollectionTree"):
    """Returns fpr, tpr, thr for two root-files containing classifier scores"""
    sig = root2array(sigf, treename=tree, branches=[branch, weight])
    bkg = root2array(bkgf, treename=tree, branches=[branch, weight])

    sig_score = sig[branchname]
    sig_weight = sig[weight]
    sig_truth = np.ones_like(sig_score)

    bkg_score = bkg[branchname]
    bkg_weight = sig[weight]
    bkg_truth = np.zeros_like(bkg_score)

    score = np.concatenate([sig_score, bkg_score])
    weight = np.concatenate([sig_weight, bkg_weight])
    truth = np.concatenate([sig_truth, bkg_truth])

    return roc_curve(truth, score, sample_weight=weight)


def bdt_threshold(tpr, thr, eff):
    """Returns BDT threshold at 'eff'-efficiency"""
    return thr[np.argmax(tpr > eff)]


def rejection(score, weight, thr):
    """Calculates the rejection for a given BDT threshold"""
    total_weight = np.sum(weight)

    pass_thr = score > thr
    pass_weight = np.sum(weight[pass_thr])
    pass_weight_sqr = np.sum(np.power(weight[pass_thr], 2))

    rej = total_weight / pass_weight

    # Rough approximation
    err = (total_weight / pass_weight) * (np.sqrt(pass_weight_sqr) / pass_weight)

    return rej, err