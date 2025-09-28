# -*- coding: utf-8 -*-
"""trauma_hemorrhage_ML_analysis.ipynb

"""
Reproduces the main analyses in Kawai et al.
Policy: the operating threshold is pre-specified on OOF to keep FNR below five percent
and is applied unchanged to the independent test set.
"""

# ===== Imports =====
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, brier_score_loss, auc
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from scipy.stats import norm

# display fallback for non-notebook environments
try:
    from IPython.display import display
except Exception:
    def display(obj):
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string(index=False))
        else:
            print(str(obj))

# ===== Global settings =====
SEED = 123
ES_ROUNDS = 200
np.random.seed(SEED)

# Threshold policy (kept fixed as in the manuscript)
USE_FIXED_THRESHOLD = True
FIXED_THRESHOLD     = 0.091

# ===== Utilities =====
def stratified_bootstrap_indices(y, n_boot=5000, seed=SEED, rng=None):
    """Class-stratified bootstrap indices. Accepts either a seed or a numpy Generator."""
    if rng is None:
        rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    out = []
    for _ in range(n_boot):
        bs_pos = rng.choice(pos, size=len(pos), replace=True)
        bs_neg = rng.choice(neg, size=len(neg), replace=True)
        out.append(np.concatenate([bs_pos, bs_neg]))
    return out

def auc_ci_boot(y, p, idx_list, alpha=0.95):
    """Bootstrap AUROC mean and percentile CI."""
    y = np.asarray(y); p = np.asarray(p)
    aucs = []
    for idx in idx_list:
        yb = y[idx]; pb = p[idx]
        if np.unique(yb).size < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))
    aucs = np.asarray(aucs, dtype=float)
    lo, hi = np.percentile(aucs, [(1-alpha)/2*100, (1+alpha)/2*100])
    return float(aucs.mean()), float(lo), float(hi)

def metrics_at_threshold(y, p, thr):
    """Sensitivity, specificity, PPV, NPV and confusion matrix entries at a threshold."""
    y = np.asarray(y); p = np.asarray(p)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) else 0.0
    npv  = tn / (tn + fn) if (tn + fn) else 0.0
    return sens, spec, ppv, npv, tn, fp, fn, tp

def metrics_ci_boot(y, p, thr, idx_list):
    """Bootstrap CIs for threshold-based metrics."""
    sens_b, spec_b, ppv_b, npv_b = [], [], [], []
    for idx in idx_list:
        yb = np.asarray(y)[idx]; pb = np.asarray(p)[idx]
        if np.unique(yb).size < 2:
            continue
        s, c, ppv, npv, *_ = metrics_at_threshold(yb, pb, thr)
        sens_b.append(s); spec_b.append(c); ppv_b.append(ppv); npv_b.append(npv)
    def ci(a):
        a = np.asarray(a, dtype=float)
        lo, hi = np.percentile(a, [2.5, 97.5])
        return float(a.mean()), float(lo), float(hi)
    return {
        "sensitivity": ci(sens_b),
        "specificity": ci(spec_b),
        "PPV": ci(ppv_b),
        "NPV": ci(npv_b),
    }

def calibration_stats(y, p):
    """Logistic recalibration on logit(p): return intercept, slope, and Brier."""
    eps = 1e-6
    p_clip = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", penalty="l2", C=1e6, max_iter=1000, random_state=SEED)
    lr.fit(logit_p, y)
    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_.ravel()[0])
    brier = float(brier_score_loss(y, p_clip))
    return intercept, slope, brier

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

# DeLong for correlated ROC curves
def _phi_matrix(pos_scores: np.ndarray, neg_scores: np.ndarray) -> np.ndarray:
    pos = pos_scores.reshape(-1, 1)
    neg = neg_scores.reshape(1, -1)
    gt = (pos > neg).astype(float)
    eq = (pos == neg).astype(float)
    return gt + 0.5 * eq

def _auc_v10_v01(y_true: np.ndarray, scores: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    m = len(pos_scores); n = len(neg_scores)
    if m < 2 or n < 2:
        raise ValueError("Need at least two positive and two negative cases.")
    phi = _phi_matrix(pos_scores, neg_scores)
    auc = phi.mean()
    V10 = phi.mean(axis=1)
    V01 = phi.mean(axis=0)
    return auc, V10, V01

def delong_test_correlated(y_true: np.ndarray, scores1: np.ndarray, scores2: np.ndarray):
    """DeLong test and CIs for two correlated ROC AUCs on the same sample."""
    y = np.asarray(y_true).astype(int)
    s1 = np.asarray(scores1, dtype=float)
    s2 = np.asarray(scores2, dtype=float)

    auc1, V10_1, V01_1 = _auc_v10_v01(y, s1)
    auc2, V10_2, V01_2 = _auc_v10_v01(y, s2)

    m = len(V10_1); n = len(V01_1)
    var1 = np.var(V10_1, ddof=1)/m + np.var(V01_1, ddof=1)/n
    var2 = np.var(V10_2, ddof=1)/m + np.var(V01_2, ddof=1)/n
    cov_V10 = np.cov(V10_1, V10_2, ddof=1)[0,1] / m
    cov_V01 = np.cov(V01_1, V01_2, ddof=1)[0,1] / n
    cov12   = cov_V10 + cov_V01

    delta = auc1 - auc2
    var_diff = max(var1 + var2 - 2.0 * cov12, 1e-12)
    z = delta / np.sqrt(var_diff)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))

    auc1_ci = (auc1 - 1.96*np.sqrt(var1),  auc1 + 1.96*np.sqrt(var1))
    auc2_ci = (auc2 - 1.96*np.sqrt(var2),  auc2 + 1.96*np.sqrt(var2))
    delta_ci = (delta - 1.96*np.sqrt(var_diff), delta + 1.96*np.sqrt(var_diff))

    return {
        "auc1": auc1, "auc1_ci": auc1_ci,
        "auc2": auc2, "auc2_ci": auc2_ci,
        "delta": delta, "delta_ci": delta_ci,
        "z": z, "p": p, "var_diff": var_diff
    }

def pick_threshold_constrained(y, p, fnr_max=0.05, fpr_max=0.50, prefer="min_fpr"):
    """Pick OOF threshold with FNR ≤ fnr_max and FPR ≤ fpr_max."""
    y = np.asarray(y); p = np.asarray(p)
    fpr, tpr, thr = roc_curve(y, p)
    fnr = 1.0 - tpr
    ok = (fnr <= fnr_max) & (fpr <= fpr_max)
    if ok.any():
        idx_ok = np.where(ok)[0]
        if prefer == "youden":
            i = idx_ok[np.argmax(tpr[idx_ok] - fpr[idx_ok])]
        else:
            i = idx_ok[np.argmin(fpr[idx_ok])]
    else:
        viol = np.square(np.clip(fnr - fnr_max, 0, None)) + np.square(np.clip(fpr - fpr_max, 0, None))
        i = int(np.argmin(viol))
    return float(thr[i])

# ===== Preprocess =====
def preprocess(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()

    drop_cols = ["受傷日","退院日","予後","氏名","疾患カテゴリ","主病名","静脈疑い","受傷時間",
                 "交通事故_相手","交通事故_本人","病着時間","救急隊現着時間","PS終了時間"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df = df.rename(columns={
        '性別２':"Sex",'年齢':"Age","常用薬":"medicine",'搬入手段':"transport",
        "Blunt_Injury_除外":"mechanism_pre",
        '現着DBP':"scene_DBP",'現着HR':"scene_HR",'現着RR':"scene_RR",'現着SBP':"scene_SBP",
        '病着DBP':"hos_DBP",'病着HR':"hos_HR",'病着RR':"hos_RR",'病着SBP':"hos_SBP",
        "BEインポート":"BE","HCO3インポート":"HCO3",
        'PS後DBP':"postPS_DBP",'PS後HR':"postPS_HR",'PS後RR':"postPS_RR",'PS後SBP':"postPS_SBP",
        "低体温":"hypothermia","切迫するD":"neuro_disability",
        '病院前LSI':"prehosLSI","来院後LSI":"hosLSI",
        "hos緊急止血術":"S+IVR"
    })

    # Mechanism extraction
    def extract_relevant_category(text):
        if pd.isna(text):
            return None
        for cat in ["交通事故","転倒","転落墜落","鋭的","その他"]:
            if cat in str(text):
                return cat
        return None
    if "mechanism_pre" in df.columns:
        df["mechanism"] = df["mechanism_pre"].apply(extract_relevant_category)
        df = df.drop(columns=["mechanism_pre"], errors="ignore")

    # Value normalization
    df = df.replace({
        "Sex":{"M":1, "Ｍ":1, "F":0, "Ｆ":0},
        "prehosFAST":{"陰性":"Neg","陽性":"Pos","未実施":"Not"},
        "hosFAST":{"陰性":"Neg","陽性":"Pos","未実施":"Not"},
        "neuro_disability":{"No":0,"Yes":1},
        "hypothermia":{"Yes":1,"No":0},
        "mechanism":{
            "交通事故":"Traffic_Accident","鋭的":"Penetrating_Trauma",
            "転倒":"Fall","転落墜落":"Fall_from_height","その他":"Other"
        },
        "transport":{"救急車":"EMS","ドクターカー":"Drcar","ドクターヘリ":"HEM","その他":"other"}
    })

    # Binary label mapping for S+IVR
    s_map = {"なし":0, "CT前Surg":0, "ECMO":0, "頭部Surg":0, "非止血整形手術":0,
             "頭部Surg\nなし":0, "なし\n頭部Surg":0, 'なし\n非止血整形手術':0,
             "なし\n非止血整形手術\n頭部Surg":0, "非止血整形手術\nなし":0,
             "体幹Surgery":1,"体幹IVR":1,"体幹Surg→IVR":1,"骨盤四肢Surg":1,
             "骨盤四肢Surg\n体幹IVR→Surg":1,"骨盤四肢Surg\n体幹Surgery":1,
             "骨盤四肢Surg\n体幹IVR":1,'体幹IVR\n骨盤四肢Surg':1,"体幹IVR→Surg":1,
             "頭部Surg\n体幹IVR":1,"体幹IVR\n非止血整形手術":1,"体幹Surgery\n骨盤四肢Surg":1,
             "非止血整形手術\n体幹IVR":1,"体幹IVR\n頭部Surg":1}
    if "S+IVR" in df.columns:
        df["S+IVR"] = df["S+IVR"].map(s_map).astype("Int64")

    # Multilabel binarization
    repl = {
        "prehosLSI":{"なし":"none","挿管":"intubate","胸腔ドレナージ":"chest_tube",
                     "タニケット":"tanicket","骨盤バインダー":"pelvic_binder","IABO":"IABO"},
        "hosLSI":{"なし":"none","挿管":"intubate","胸腔ドレナージ":"chest_tube",
                  "タニケット":"tanicket","骨盤バインダー":"pelvic_binder","IABO":"IABO"},
        "medicine":{"DM薬":"DM","抗血栓薬":"anticoag","該当なし":"none","βブロッカ":"beta"}
    }
    for col in ["prehosLSI","hosLSI","medicine"]:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(
                lambda x: [repl[col].get(item, item) for item in str(x).split("\n")]
            )
            df[col] = df[col].apply(set)
            mlb = MultiLabelBinarizer()
            enc = pd.DataFrame(mlb.fit_transform(df[col]),
                               columns=[f"{col}_{c}" for c in mlb.classes_],
                               index=df.index)
            df = pd.concat([df, enc], axis=1)
    df = df.drop(columns=[c for c in ["prehosLSI","hosLSI","medicine"] if c in df.columns], errors="ignore")

    # FAST two-level representation
    if "prehosFAST" in df.columns:
        df["prehosFAST_performed"] = (df["prehosFAST"] != "Not").astype(int)
        df["prehosFAST_pos"]       = (df["prehosFAST"] == "Pos").astype(int)
    if "hosFAST" in df.columns:
        df["hosFAST_performed"] = (df["hosFAST"] != "Not").astype(int)
        df["hosFAST_pos"]       = (df["hosFAST"] == "Pos").astype(int)

    # One-hot for transport and mechanism
    for cat_col in ["transport","mechanism"]:
        if cat_col in df.columns:
            df = pd.concat([df, pd.get_dummies(df[cat_col], prefix=cat_col, dtype=int)], axis=1)
    df = df.drop(columns=[c for c in ["transport","prehosFAST","hosFAST","mechanism"] if c in df.columns],
                 errors="ignore")

    # SI and deltas
    needed = {"time_difference","PS_interval"}
    if not needed.issubset(df.columns):
        missing = list(needed - set(df.columns))
        raise ValueError(f"Required time columns are missing: {missing}")

    for a,b,out in [("scene_HR","scene_SBP","scene_SI"),
                    ("hos_HR","hos_SBP","hos_SI"),
                    ("postPS_HR","postPS_SBP","postPS_SI")]:
        if a in df.columns and b in df.columns:
            df[out] = df[a] / df[b]

    eps1 = df["time_difference"].clip(lower=1.0)
    eps2 = df["PS_interval"].clip(lower=1.0)

    df["HR_delta_1"]  = (df["hos_HR"]    - df["scene_HR"])  / eps1
    df["SBP_delta_1"] = (df["hos_SBP"]   - df["scene_SBP"]) / eps1
    df["DBP_delta_1"] = (df["hos_DBP"]   - df["scene_DBP"]) / eps1
    df["RR_delta_1"]  = (df["hos_RR"]    - df["scene_RR"])  / eps1
    df["SI_delta_1"]  = (df["hos_SI"]    - df["scene_SI"])  / eps1

    df["HR_delta_2"]  = (df["postPS_HR"]   - df["hos_HR"])  / eps2
    df["SBP_delta_2"] = (df["postPS_SBP"]  - df["hos_SBP"]) / eps2
    df["DBP_delta_2"] = (df["postPS_DBP"]  - df["hos_DBP"]) / eps2
    df["RR_delta_2"]  = (df["postPS_RR"]   - df["hos_RR"])  / eps2
    df["SI_delta_2"]  = (df["postPS_SI"]   - df["hos_SI"])  / eps2

    for c in ["HR_delta_1","SBP_delta_1","DBP_delta_1","RR_delta_1","SI_delta_1",
              "HR_delta_2","SBP_delta_2","DBP_delta_2","RR_delta_2","SI_delta_2"]:
        if c in df.columns:
            q1, q99 = df[c].quantile([0.01, 0.99])
            df[c] = df[c].clip(lower=q1, upper=q99)

    df = df.drop(columns=[c for c in ["time_difference","PS_interval","PS後SI"] if c in df.columns],
                 errors="ignore")

    # finalize label and dtypes
    df = df.dropna(subset=["S+IVR"])
    df["S+IVR"] = df["S+IVR"].astype(int)
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(int)
    return df

def standardize_train_test(x_train, x_test):
    """Standardize continuous fields used by LR baseline and for stability."""
    scaling_columns = [
        'Age','BE','HCO3','Lac','pH','Hb',
        'postPS_DBP','postPS_HR','postPS_RR','postPS_SBP',
        'scene_DBP','scene_HR','scene_RR','scene_SBP',
        'hos_DBP','hos_HR','hos_RR','hos_SBP',
        "HR_delta_1","SBP_delta_1","DBP_delta_1","RR_delta_1","SI_delta_1",
        "HR_delta_2","SBP_delta_2","DBP_delta_2","RR_delta_2","SI_delta_2",
        "scene_SI","hos_SI","postPS_SI"
    ]
    cols = [c for c in scaling_columns if c in x_train.columns]
    sc = StandardScaler().fit(x_train[cols])
    x_train[cols] = sc.transform(x_train[cols])
    x_test[cols]  = sc.transform(x_test[cols])
    return x_train, x_test

# ===== LightGBM tuning, training, and calibration =====
def tune_lgbm(x_train, y_train, cv_splits):
    fixed_params = {
        "boosting_type":"gbdt","objective":"binary","metric":"auc",
        "verbosity":-1,"n_estimators":5000,"seed":SEED,
        "bagging_seed":SEED,"feature_fraction_seed":SEED,"n_jobs":1
    }
    def objective(trial):
        pos = int(y_train.sum()); neg = len(y_train) - pos
        base_spw = neg / max(pos, 1)
        low  = max(1.0, base_spw * 0.33)
        high = base_spw * 3.0
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.06),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", -1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 60),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-6, 1e-3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.80, 1.00),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.80, 1.00),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 3),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", low, high, log=True),
            "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 20.0),
        }
        params = {**fixed_params, **params}
        aucs = []
        for tr_idx, va_idx in cv_splits:
            x_tr, y_tr = x_train.iloc[tr_idx], y_train.iloc[tr_idx]
            x_va, y_va = x_train.iloc[va_idx], y_train.iloc[va_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)],
                      callbacks=[lgb.early_stopping(stopping_rounds=ES_ROUNDS, verbose=-1)])
            y_va_pred = model.predict_proba(x_va)[:, 1]
            aucs.append(roc_auc_score(y_va, y_va_pred))
        return float(np.mean(aucs))
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    best_params = study.best_params
    return {**fixed_params, **best_params}

def train_lgbm_and_predict(x_train, y_train, x_test, params, cv_splits):
    """Return OOF probabilities and mean test probabilities across folds."""
    oof_pred = np.zeros(y_train.shape[0], dtype=float)
    test_pred_folds = np.zeros((x_test.shape[0], len(cv_splits)), dtype=float)
    val_auc_scores = []
    for k, (tr_idx, va_idx) in enumerate(cv_splits):
        x_tr, y_tr = x_train.iloc[tr_idx], y_train.iloc[tr_idx]
        x_va, y_va = x_train.iloc[va_idx], y_train.iloc[va_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)],
                  callbacks=[lgb.early_stopping(stopping_rounds=ES_ROUNDS, verbose=-1)])
        oof_pred[va_idx] = model.predict_proba(x_va)[:, 1]
        test_pred_folds[:, k] = model.predict_proba(x_test)[:, 1]
        val_auc_scores.append(roc_auc_score(y_va, oof_pred[va_idx]))
    test_pred_mean = test_pred_folds.mean(axis=1)
    return oof_pred, test_pred_mean, val_auc_scores

def platt_recalibration(p_train, y_train, p_test):
    """Platt-type logistic recalibration on logit probabilities."""
    lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000, random_state=SEED)
    lr.fit(logit(p_train).reshape(-1,1), y_train)
    p_train_cal = lr.predict_proba(logit(p_train).reshape(-1,1))[:,1]
    p_test_cal  = lr.predict_proba(logit(p_test).reshape(-1,1))[:,1]
    return p_train_cal, p_test_cal

# ===== Logistic regression baseline =====
def tune_fit_logreg(x_train, y_train, cv_splits):
    def make_lr(params):
        C = params["C"]; penalty = params["penalty"]
        if penalty == "elasticnet":
            return LogisticRegression(C=C, penalty="elasticnet", l1_ratio=params["l1_ratio"],
                                      solver="saga", max_iter=1000, random_state=SEED)
        elif penalty == "l1":
            return LogisticRegression(C=C, penalty="l1", solver="liblinear",
                                      max_iter=1000, random_state=SEED)
        elif penalty == "l2":
            return LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                      max_iter=1000, random_state=SEED)
        else:
            return LogisticRegression(penalty=None, solver="lbfgs",
                                      max_iter=1000, random_state=SEED)

    def objective(trial):
        C = trial.suggest_float("C", 1e-6, 1e3, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1","l2","elasticnet", None])
        params = {"C":C, "penalty":penalty}
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        model = make_lr(params)
        scores = cross_val_score(model, x_train, y_train, cv=list(cv_splits), scoring="roc_auc")
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    best = study.best_params
    if "l1_ratio" not in best:
        best["l1_ratio"] = None
    model = make_lr(best)
    model.fit(x_train, y_train)
    return model

# ===== Plots =====
def show_roc_plot(y_test, p_lgbm, p_lr):
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, p_lgbm)
    fpr_lr,  tpr_lr,  _ = roc_curve(y_test, p_lr)
    roc_auc_lgb = auc(fpr_lgb, tpr_lgb)
    roc_auc_lr  = auc(fpr_lr, tpr_lr)
    tpr_at05_lgb = np.interp(0.5, fpr_lgb, tpr_lgb)
    tpr_at05_lr  = np.interp(0.5, fpr_lr,  tpr_lr)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(fpr_lgb, tpr_lgb, lw=2, label=f'LightGBM AUC {roc_auc_lgb:.2f}')
    ax.plot(fpr_lr,  tpr_lr,  lw=2, label=f'Logistic AUC {roc_auc_lr:.2f}')
    ax.plot([0,1],[0,1], linestyle="--", color="gray", lw=1)
    ax.plot([0.5,0.5],[0,tpr_at05_lgb], color='C0', linestyle=':', lw=1)
    ax.plot([0,0.5],[tpr_at05_lgb,tpr_at05_lgb], color='C0', linestyle=':', lw=1)
    ax.plot([0.5,0.5],[0,tpr_at05_lr],  color='C1', linestyle=':', lw=1)
    ax.plot([0,0.5],[tpr_at05_lr, tpr_at05_lr],  color='C1', linestyle=':', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves on the independent test set")
    ax.legend(loc="lower right")
    plt.show()

def show_calibration_plot(y_test, p_pre, p_post):
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, p_pre, n_bins=10, strategy="quantile")
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, p_post, n_bins=10, strategy="quantile")
    cal_int_pre,  cal_slope_pre,  brier_pre  = calibration_stats(y_test, p_pre)
    cal_int_post, cal_slope_post, brier_post = calibration_stats(y_test, p_post)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot([0,1],[0,1], linestyle="--", lw=1.2, label="Perfect calibration")
    ax.plot(prob_pred_raw, prob_true_raw, marker="o", lw=1.5, label="Pre calibration")
    ax.plot(prob_pred_cal, prob_true_cal, marker="s", lw=1.5, label="Post calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration on the independent test set")
    ax.legend(loc="lower right")
    text_str = (
        f"Pre  Intercept {cal_int_pre:.2f}, Slope {cal_slope_pre:.2f}, Brier {brier_pre:.3f}\n"
        f"Post Intercept {cal_int_post:.2f}, Slope {cal_slope_post:.2f}, Brier {brier_post:.3f}"
    )
    ax.text(0.05, 0.78, text_str, transform=ax.transAxes)
    plt.show()

def append_row(rows, dataset, model, thr, auc_mean, auc_lo, auc_hi, m_dict, cal_tuple):
    rows.append({
        "Dataset": dataset,
        "Model": model,
        "Threshold": thr,
        "AUROC": round(auc_mean, 3),
        "AUROC_CI_low": round(auc_lo, 3),
        "AUROC_CI_high": round(auc_hi, 3),
        "Sensitivity_mean": round(m_dict["sensitivity"][0], 3),
        "Sensitivity_CI_low": round(m_dict["sensitivity"][1], 3),
        "Sensitivity_CI_high": round(m_dict["sensitivity"][2], 3),
        "Specificity_mean": round(m_dict["specificity"][0], 3),
        "Specificity_CI_low": round(m_dict["specificity"][1], 3),
        "Specificity_CI_high": round(m_dict["specificity"][2], 3),
        "PPV_mean": round(m_dict["PPV"][0], 3),
        "PPV_CI_low": round(m_dict["PPV"][1], 3),
        "PPV_CI_high": round(m_dict["PPV"][2], 3),
        "NPV_mean": round(m_dict["NPV"][0], 3),
        "NPV_CI_low": round(m_dict["NPV"][1], 3),
        "NPV_CI_high": round(m_dict["NPV"][2], 3),
        "Calib_intercept": round(cal_tuple[0], 3),
        "Calib_slope": round(cal_tuple[1], 3),
        "Brier": round(cal_tuple[2], 3),
    })

# ===== Run: main analysis =====
if "df_all" not in globals():
    raise RuntimeError("Please load your dataset into a pandas DataFrame named `df_all` before running.")

# Preprocess
df = preprocess(df_all)

# Label and features
y = df["S+IVR"].astype(int)
X = df.drop(columns=["S+IVR"])
if "ID" in X.columns:
    X = X.drop(columns=["ID"])

# 80:20 stratified split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=SEED)

# Standardize continuous variables
X_tr, X_te = standardize_train_test(X_tr.copy(), X_te.copy())

# Aliases (ensure later blocks use the standardized matrices)
x_train, y_train = X_tr, y_tr
x_test,  y_test  = X_te, y_te

# CV folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_splits = list(cv.split(X_tr, y_tr))

# LightGBM tuning and training
lgbm_params = tune_lgbm(X_tr, y_tr, cv_splits)
oof_lgbm_raw, te_lgbm_raw, _ = train_lgbm_and_predict(X_tr, y_tr, X_te, lgbm_params, cv_splits)

# Platt-type recalibration on logit probabilities
oof_lgbm_cal, te_lgbm_cal = platt_recalibration(oof_lgbm_raw, y_tr, te_lgbm_raw)

# Logistic regression baseline
lr_model = tune_fit_logreg(X_tr, y_tr, cv_splits)
te_lr = lr_model.predict_proba(X_te)[:, 1]

# AUROC with bootstrap CI
idx_cv = stratified_bootstrap_indices(y_tr, n_boot=5000, seed=SEED)
idx_te = stratified_bootstrap_indices(y_te, n_boot=5000, seed=SEED)
auc_cv_lgbm, lo_cv_lgbm, hi_cv_lgbm = auc_ci_boot(y_tr, oof_lgbm_cal, idx_cv)
auc_te_lgbm, lo_te_lgbm, hi_te_lgbm = auc_ci_boot(y_te, te_lgbm_cal, idx_te)
auc_te_lr,   lo_te_lr,   hi_te_lr   = auc_ci_boot(y_te, te_lr,       idx_te)

print(f"[LightGBM] Test AUROC {auc_te_lgbm:.3f} [{lo_te_lgbm:.3f}, {hi_te_lgbm:.3f}]")
print(f"[Logistic ] Test AUROC {auc_te_lr:.3f} [{lo_te_lr:.3f}, {hi_te_lr:.3f}]")

# DeLong test on the same test set
res = delong_test_correlated(y_te, te_lgbm_cal, te_lr)
delta = res["delta"]
se    = np.sqrt(res["var_diff"])
ci_lo = delta - 1.96*se
ci_hi = delta + 1.96*se
print(f"[DeLong] ΔAUC {delta:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={res['p']:.3f}")
print(f"[DeLong] LGBM AUC {res['auc1']:.3f} [{res['auc1_ci'][0]:.3f}, {res['auc1_ci'][1]:.3f}]")
print(f"[DeLong]  LR  AUC {res['auc2']:.3f} [{res['auc2_ci'][0]:.3f}, {res['auc2_ci'][1]:.3f}]")

# Choose operating threshold
thr_use = FIXED_THRESHOLD if USE_FIXED_THRESHOLD else \
          pick_threshold_constrained(y_tr, oof_lgbm_cal, fnr_max=0.05, fpr_max=0.50, prefer="min_fpr")
print(f"[Threshold] Operating threshold used: {thr_use:.3f}")

# Metrics and calibration at the chosen threshold
m_cv_lgbm = metrics_ci_boot(y_tr, oof_lgbm_cal, thr_use, idx_cv)
m_te_lgbm = metrics_ci_boot(y_te, te_lgbm_cal, thr_use, idx_te)
m_te_lr   = metrics_ci_boot(y_te, te_lr,        thr_use, idx_te)

cal_cv_lgbm = calibration_stats(y_tr, oof_lgbm_cal)
cal_te_lgbm = calibration_stats(y_te, te_lgbm_cal)
cal_te_lr   = calibration_stats(y_te, te_lr)

# Performance summary table (display only)
rows = []
append_row(rows, "Train or validation (OOF)", "LightGBM", thr_use,
           auc_cv_lgbm, lo_cv_lgbm, hi_cv_lgbm, m_cv_lgbm, cal_cv_lgbm)
append_row(rows, "Independent test", "LightGBM", thr_use,
           auc_te_lgbm, lo_te_lgbm, hi_te_lgbm, m_te_lgbm, cal_te_lgbm)
append_row(rows, "Independent test", "Logistic regression", thr_use,
           auc_te_lr, lo_te_lr, hi_te_lr, m_te_lr, cal_te_lr)
perf_df = pd.DataFrame(rows)
display(perf_df)

# Calibration plot and ROC curves
show_calibration_plot(y_te, te_lgbm_raw, te_lgbm_cal)
show_roc_plot(y_te, te_lgbm_cal, te_lr)

# Confusion matrix at the operating threshold
sens, spec, ppv, npv, tn, fp, fn, tp = metrics_at_threshold(y_te, te_lgbm_cal, thr_use)
print("\nConfusion matrix at threshold", thr_use)
print(f"TP {tp}  FN {fn}  FP {fp}  TN {tn}")
print(f"Sensitivity {sens:.3f}  Specificity {spec:.3f}  PPV {ppv:.3f}  NPV {npv:.3f}")

# ===== Decision curve analysis (DCA) =====
def net_benefit(y_true, p, thresholds, per100=False):
    """Net benefit at each threshold (per patient by default)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    N = len(y)
    nb = []
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        w = pt / (1.0 - pt)
        val = (tp / N) - (fp / N) * w
        nb.append(val * (100 if per100 else 1))
    return np.array(nb)

def dca_curves(y_true, p, thresholds, per100=False):
    """Return model, Treat-all, and Treat-none net benefit curves."""
    prev = float(np.mean(y_true))
    model = net_benefit(y_true, p, thresholds, per100=per100)
    treat_all = (prev - (1 - prev) * (thresholds / (1 - thresholds))) * (100 if per100 else 1)
    treat_none = np.zeros_like(thresholds)
    return model, treat_all, treat_none

def dca_plot(y_true, p, thr_mark=None, title="Decision curve (independent test set)",
             per100=False, bootstrap_ci=True, n_boot=2000, seed=SEED):
    """Plot DCA with optional 95% bootstrap CI."""
    rng = np.random.default_rng(seed)
    thresholds = np.arange(0.001, 0.501, 0.001)
    model, allc, none = dca_curves(y_true, p, thresholds, per100=per100)

    plt.figure(figsize=(7.0, 5.5))
    plt.plot(thresholds, model, lw=1.8, label="Model")
    plt.plot(thresholds, allc,  '--', lw=1.2, color='gray', label="Treat all")
    plt.plot(thresholds, none,  ':',  lw=1.2, color='gray', label="Treat none")

    if bootstrap_ci:
        y_arr = np.asarray(y_true, dtype=int)
        p_arr = np.asarray(p, dtype=float)
        pos = np.where(y_arr == 1)[0]
        neg = np.where(y_arr == 0)[0]
        band = []
        for _ in range(n_boot):
            bs_pos = rng.choice(pos, size=len(pos), replace=True)
            bs_neg = rng.choice(neg, size=len(neg), replace=True)
            idx = np.concatenate([bs_pos, bs_neg])
            band.append(net_benefit(y_arr[idx], p_arr[idx], thresholds, per100=per100))
        band = np.stack(band, axis=0)
        lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
        plt.fill_between(thresholds, lo, hi, alpha=0.18, label="Model 95% CI")

    if thr_mark is not None:
        plt.axvline(thr_mark, color='red', ls='--', lw=1.6,
                    label=f"Operating threshold ({thr_mark:.3f})")

    plt.axhline(0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    plt.xlim(0.00, 0.50)
    plt.xticks(np.arange(0.00, 0.51, 0.05))
    plt.xlabel("Threshold probability (pt)")
    plt.ylabel("Net benefit" + (" (per 100 patients)" if per100 else " (per patient)"))
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Run DCA on LightGBM test predictions
dca_plot(y_true=y_te, p=te_lgbm_cal, thr_mark=thr_use,
         title="Decision curve analysis on the independent test set",
         per100=False, bootstrap_ci=True, n_boot=2000, seed=SEED)

# ===== Forward selection and AUC vs feature count =====
def inner_cv_auc_lgbm(X, y, lgb_params, cv):
    """Mean AUROC via inner CV using LGBM with early stopping."""
    aucs = []
    for tr, va in cv.split(X, y):
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X.iloc[tr], y.iloc[tr],
                  eval_set=[(X.iloc[va], y.iloc[va])],
                  callbacks=[lgb.early_stopping(stopping_rounds=ES_ROUNDS, verbose=-1)])
        p_va = model.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], p_va))
    return float(np.mean(aucs))

def forward_path_auc(X, y, lgb_params, cv, max_features=None, min_gain=0.001):
    """Greedy forward selection; stop when marginal gain falls below min_gain."""
    remaining = list(X.columns)
    selected, history = [], []
    current_mean = 0.5
    while remaining and (max_features is None or len(selected) < max_features):
        best_feat, best_mean = None, -np.inf
        for f in remaining:
            feats = selected + [f]
            m = inner_cv_auc_lgbm(X[feats], y, lgb_params, cv)
            if m > best_mean:
                best_mean, best_feat = m, f
        gain = best_mean - current_mean
        selected.append(best_feat)
        remaining.remove(best_feat)
        # estimate SE across folds for display
        fold_scores = []
        for tr, va in cv.split(X[selected], y):
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(X[selected].iloc[tr], y.iloc[tr],
                      eval_set=[(X[selected].iloc[va], y.iloc[va])],
                      callbacks=[lgb.early_stopping(stopping_rounds=ES_ROUNDS, verbose=-1)])
            p_va = model.predict_proba(X[selected].iloc[va])[:, 1]
            fold_scores.append(roc_auc_score(y.iloc[va], p_va))
        fold_scores = np.array(fold_scores, dtype=float)
        se = fold_scores.std(ddof=1) / np.sqrt(cv.get_n_splits())
        history.append({
            "k": len(selected),
            "added": best_feat,
            "cv_mean_auc": best_mean,
            "cv_se": float(se),
            "feats": selected.copy()
        })
        current_mean = best_mean
        if gain < min_gain:
            break
    return history

def pick_1se_subset(history):
    """1-SE rule: smallest k with mean ≥ (max_mean - max_se)."""
    means = np.array([h["cv_mean_auc"] for h in history])
    ses   = np.array([h["cv_se"]       for h in history])
    k_best = int(np.argmax(means))
    threshold = means[k_best] - ses[k_best]
    for h in history:
        if h["cv_mean_auc"] >= threshold:
            return h
    return history[k_best]

def train_cv_ensemble_and_get_auc(
    X_train, y_train, X_test, y_test, lgb_params, cv_splits,
    agg="weighted_trimmed", alpha=0.5, seed=SEED
):
    """
    Final pipeline mirroring the main analysis:
      fold LightGBM -> meta logistic regression on OOF -> aggregate fold test probs.
    """
    n_folds = len(cv_splits)
    val_pred = np.zeros(len(X_train), dtype=float)
    val_true = np.zeros(len(X_train), dtype=np.int8)
    test_each = np.zeros((len(X_test), n_folds), dtype=float)
    val_auc_scores = []

    for k, (tr, va) in enumerate(cv_splits):
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train.iloc[tr], y_train.iloc[tr],
                  eval_set=[(X_train.iloc[va], y_train.iloc[va])],
                  callbacks=[lgb.early_stopping(stopping_rounds=ES_ROUNDS, verbose=-1)])
        p_va = model.predict_proba(X_train.iloc[va])[:, 1]
        val_pred[va] = p_va
        val_true[va] = y_train.iloc[va].values
        test_each[:, k] = model.predict_proba(X_test)[:, 1]
        val_auc_scores.append(roc_auc_score(y_train.iloc[va], p_va))

    meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    meta.fit(val_pred.reshape(-1, 1), val_true)

    if agg == "mean":
        agg_test = test_each.mean(axis=1)
    elif agg == "weighted_trimmed":
        w = np.array(val_auc_scores, dtype=float); w = w / (w.sum() if w.sum() != 0 else 1.0)
        P = test_each
        imin = np.argmin(P, axis=1); imax = np.argmax(P, axis=1)
        mask = np.ones_like(P, dtype=bool); rows = np.arange(P.shape[0])
        mask[rows, imin] = False; mask[rows, imax] = False
        W = np.broadcast_to(w.reshape(1, -1), P.shape)
        Wm = np.where(mask, W, 0.0)
        Wm = Wm / (Wm.sum(axis=1, keepdims=True) + 1e-12)
        p_trim = (P * Wm).sum(axis=1)
        p_mean = np.average(P, axis=1, weights=w)
        agg_test = alpha * p_mean + (1 - alpha) * p_trim
    else:
        raise ValueError("agg must be 'mean' or 'weighted_trimmed'")

    p_test = meta.predict_proba(agg_test.reshape(-1, 1))[:, 1]
    auc_val = roc_auc_score(y_test, p_test)
    return auc_val, p_test

# Build forward path on training set
inner_cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
outer_cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_splits_fs = list(outer_cv.split(x_train, y_train))

history = forward_path_auc(x_train, y_train, lgb_params=lgbm_params, cv=inner_cv,
                           max_features=None, min_gain=0.001)
sol_1se = pick_1se_subset(history)
print(f"1-SE solution: k={sol_1se['k']}, CV mean AUC={sol_1se['cv_mean_auc']:.3f}±{sol_1se['cv_se']:.3f}")
print("Selected features:", sol_1se["feats"])

# Evaluate along the path on the held-out test set
rows = []
for h in history:
    feats = h["feats"]
    auc_k, _ = train_cv_ensemble_and_get_auc(
        x_train[feats], y_train, x_test[feats], y_test,
        lgb_params=lgbm_params, cv_splits=cv_splits_fs,
        agg="weighted_trimmed", alpha=0.5
    )
    rows.append({"top_n": h["k"], "AUC": auc_k, "feature_names": feats})
df_path = pd.DataFrame(rows)
print(df_path.head(60))

# Plot performance vs feature count
fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=150)
ax.plot(df_path["top_n"], df_path["AUC"], marker="o")
ax.set_xlabel("Number of features (forward path)")
ax.set_ylabel("Test AUROC")
ax.set_title("Performance vs feature count (nested forward selection)")
ax.grid(True, alpha=.3)
fig.tight_layout()
plt.show()

# ===== Unified comparison: Full-63 vs Top-6 vs 1-SE (same pipeline) =====
if len(history) < 6:
    raise RuntimeError("Forward path has fewer than 6 steps. Cannot form Top-6 subset.")
feats_6  = history[5]["feats"]
feats_1s = sol_1se["feats"]

auc_full, p_full = train_cv_ensemble_and_get_auc(
    x_train, y_train, x_test, y_test,
    lgb_params=lgbm_params, cv_splits=cv_splits_fs
)
auc_6, p_6 = train_cv_ensemble_and_get_auc(
    x_train[feats_6], y_train, x_test[feats_6], y_test,
    lgb_params=lgbm_params, cv_splits=cv_splits_fs
)
auc_1s, p_1s = train_cv_ensemble_and_get_auc(
    x_train[feats_1s], y_train, x_test[feats_1s], y_test,
    lgb_params=lgbm_params, cv_splits=cv_splits_fs
)

print(f"[Full-63] Test AUROC = {auc_full:.4f}")
print(f"[Top-6 ] Test AUROC = {auc_6:.4f}")
print(f"[1-SE  ] Test AUROC = {auc_1s:.4f}")

# Bootstrap CIs and paired differences
rng = np.random.default_rng(12345)
idx_list = stratified_bootstrap_indices(y_test, n_boot=5000, rng=rng)
mean_full, lo_full, hi_full = auc_ci_boot(y_test, p_full, idx_list)
mean_6,    lo_6,    hi_6    = auc_ci_boot(y_test, p_6,    idx_list)
mean_1s,   lo_1s,   hi_1s   = auc_ci_boot(y_test, p_1s,   idx_list)
print(f"[Full-63] bootstrap AUROC = {mean_full:.4f} [{lo_full:.4f}, {hi_full:.4f}]")
print(f"[Top-6 ] bootstrap AUROC = {mean_6:.4f} [{lo_6:.4f}, {hi_6:.4f}]")
print(f"[1-SE  ] bootstrap AUROC = {mean_1s:.4f} [{lo_1s:.4f}, {hi_1s:.4f}]")

def paired_bootstrap_auc_diff(y, s1, s2, idx_list):
    y = np.asarray(y); s1 = np.asarray(s1); s2 = np.asarray(s2)
    diffs = []
    for idx in idx_list:
        yb = y[idx]; s1b = s1[idx]; s2b = s2[idx]
        if np.unique(yb).size < 2:
            continue
        diffs.append(roc_auc_score(yb, s1b) - roc_auc_score(yb, s2b))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), (float(lo), float(hi)), float(min(max(p_two,0.0),1.0))

d_mean_6,  d_ci_6,  p_boot_6  = paired_bootstrap_auc_diff(y_test, p_full, p_6,  idx_list)
d_mean_1s, d_ci_1s, p_boot_1s = paired_bootstrap_auc_diff(y_test, p_full, p_1s, idx_list)
print(f"[Δ Top-6 vs Full]  bootstrap Δ={d_mean_6:.4f} [{d_ci_6[0]:.4f}, {d_ci_6[1]:.4f}], p_boot={p_boot_6:.4f}")
print(f"[Δ 1-SE  vs Full]  bootstrap Δ={d_mean_1s:.4f} [{d_ci_1s[0]:.4f}, {d_ci_1s[1]:.4f}], p_boot={p_boot_1s:.4f}")

# DeLong comparisons
dl_6  = delong_test_correlated(y_test, p_full, p_6)
dl_1s = delong_test_correlated(y_test, p_full, p_1s)
print(f"[DeLong 6 vs 63]  ΔAUC={dl_6['delta']:.4f} [{dl_6['delta_ci'][0]:.4f}, {dl_6['delta_ci'][1]:.4f}], p={dl_6['p']:.4f}")
print(f"  Full-63 AUC {dl_6['auc1']:.4f} [{dl_6['auc1_ci'][0]:.4f}, {dl_6['auc1_ci'][1]:.4f}]")
print(f"  Top-6  AUC {dl_6['auc2']:.4f} [{dl_6['auc2_ci'][0]:.4f}, {dl_6['auc2_ci'][1]:.4f}]")
print(f"[DeLong 1-SE vs 63] ΔAUC={dl_1s['delta']:.4f} [{dl_1s['delta_ci'][0]:.4f}, {dl_1s['delta_ci'][1]:.4f}], p={dl_1s['p']:.4f}")
print(f"  1-SE  AUC {dl_1s['auc2']:.4f} [{dl_1s['auc2_ci'][0]:.4f}, {dl_1s['auc2_ci'][1]:.4f}]")

# ===== SHAP beeswarm for 5 folds (no saving, display only) =====
import shap
def plot_shap_beeswarm_per_fold(X_tr, y_tr, cv_splits, lgbm_params, max_display=20):
    for fold_count, (tr_idx, va_idx) in enumerate(cv_splits, start=1):
        X_tr_fold, y_tr_fold = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        X_va_fold, y_va_fold = X_tr.iloc[va_idx], y_tr.iloc[va_idx]
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(X_tr_fold, y_tr_fold,
                  eval_set=[(X_va_fold, y_va_fold)],
                  callbacks=[lgb.early_stopping(stopping_rounds=ES_ROUNDS, verbose=-1)])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_va_fold)
        shap_to_plot = shap_values[1] if isinstance(shap_values, list) and len(shap_values) >= 2 else shap_values
        plt.figure(figsize=(7, 9))
        shap.summary_plot(shap_to_plot, X_va_fold, plot_type="dot", show=False, max_display=max_display)
        ax = plt.gca()
        ax.set_title(f"SHAP beeswarm (validation of fold {fold_count})", pad=10)
        plt.tight_layout()
        plt.show()

# show 5 figures
plot_shap_beeswarm_per_fold(X_tr, y_tr, cv_splits, lgbm_params, max_display=20)