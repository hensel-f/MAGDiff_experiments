import numpy as np
from scipy.stats import ks_2samp, norm

def tests_conf_interval(tests, alpha = 0.05):
    n = len(tests)
    p = np.sum(tests)/n
    if p==0:
        lerr = 0
        uerr = 3/n
    elif p==1:
        lerr = 3/n
        uerr = 0
    else:
        lerr = norm.ppf(1-alpha/2)*np.sqrt( p*(1-p) / n  )
        uerr = lerr
    return p, lerr, uerr


def one_dimensional_tests(X_tr, X_te):
    p_vals = []

    # For each dimension we conduct a separate KS test
    for i in range(X_tr.shape[1]):
        feature_tr = X_tr[:, i]
        feature_te = X_te[:, i]

        # Compute KS statistic and p-value
        t_val, p_val = ks_2samp(feature_tr, feature_te)

        p_vals.append(p_val)

    p_vals = np.array(p_vals)

    return p_vals

def BBSD(model, data_tr, data_te, alpha=.05, get_decisions_by_dim=False):

    #extract softmax output from the neural network model
    X_tr = model.predict(data_tr)
    X_te = model.predict(data_te)

    #perform the KS test for each dimension
    p_vals = one_dimensional_tests(X_tr, X_te)
    p_val = min(np.min(p_vals), 1.0)

    #get the number of single test performed
    n_KStest = X_tr.shape[1]

    #conclude using the Bonferroni correction, ensuring a control of the family-wise error rate.
    decision = (p_val < alpha/n_KStest)
    decisions_by_dim = (p_vals < np.ones(n_KStest)*alpha/n_KStest)

    if get_decisions_by_dim:
        return decision, decisions_by_dim
    else:
        return decision

def BBSD_from_pred(X_tr, X_te, alpha=.05, get_decisions_by_dim=False):

    #extract softmax output from the neural network model
#     X_tr = model.predict(data_tr)
#     X_te = model.predict(data_te)

    #perform the KS test for each dimension
    p_vals = one_dimensional_tests(X_tr, X_te)
    p_val = min(np.min(p_vals), 1.0)

    #get the number of single tests performed
    n_KStest = X_tr.shape[1]

    #conclude using the Bonferroni correction, ensuring a control of the family-wise error rate.
    decision = (p_val < alpha/n_KStest)
    decisions_by_dim = (p_vals < np.ones(n_KStest)*alpha/n_KStest)

    if get_decisions_by_dim:
        return decision, decisions_by_dim
    else:
        return decision, p_val * n_KStest

