import json, warnings, time, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, train_test_split,
                                     GridSearchCV, learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             brier_score_loss, confusion_matrix, roc_curve, matthews_corrcoef)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import calibration_curve
from sklearn.base import clone

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out") + os.sep
import os; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "figure.dpi": 110, "savefig.dpi": 130})
SEED = 42
R = {}

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
# recode: 1 = malignant (positive class), 0 = benign
y = pd.Series(1 - data.target, name="malignant")
R["n"] = len(X); R["p"] = X.shape[1]
R["n_mal"] = int(y.sum()); R["n_ben"] = int((1 - y).sum())
R["missing"] = int(X.isna().sum().sum()); R["dups"] = int(X.duplicated().sum())

# ---------- descriptive ----------
desc = X.describe().T[["mean", "std", "min", "max"]]
grp = X.groupby(y).mean().T
grp.columns = ["benign_mean", "malignant_mean"]
desc = desc.join(grp)
desc.round(4).to_csv(OUT + "descriptive.csv")

# Mann-Whitney U per feature (H4)
mw = []
for c in X.columns:
    u, pv = stats.mannwhitneyu(X.loc[y == 1, c], X.loc[y == 0, c], alternative="two-sided")
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    r_rb = 1 - 2 * u / (n1 * n0)  # rank-biserial (sign: negative => malignant higher)
    mw.append([c, u, pv, abs(r_rb)])
mw = pd.DataFrame(mw, columns=["feature", "U", "p", "effect_r"]).sort_values("effect_r", ascending=False)
mw["p_bonf"] = np.minimum(mw.p * len(X.columns), 1)
mw.to_csv(OUT + "mannwhitney.csv", index=False)
R["mw_sig"] = int((mw.p_bonf < 0.05).sum())
R["mw_nonsig"] = mw.loc[mw.p_bonf >= 0.05, "feature"].tolist()

# figure: class distribution
fig, ax = plt.subplots(figsize=(4.5, 3))
ax.bar(["Benign", "Malignant"], [R["n_ben"], R["n_mal"]], color=["#4C72B0", "#C44E52"])
for i, v in enumerate([R["n_ben"], R["n_mal"]]):
    ax.text(i, v + 5, f"{v} ({v / R['n'] * 100:.1f}%)", ha="center")
ax.set_ylabel("Number of cases"); ax.set_ylim(0, 420); ax.set_title("Class distribution")
fig.tight_layout(); fig.savefig(OUT + "fig_class.png"); plt.close()

# correlation heatmap (mean features)
mean_cols = [c for c in X.columns if c.startswith("mean")]
fig, ax = plt.subplots(figsize=(6.2, 5))
sns.heatmap(X[mean_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 7}, cbar_kws={"shrink": .8})
ax.set_title("Pearson correlation among the ten 'mean' features")
fig.tight_layout(); fig.savefig(OUT + "fig_corr.png"); plt.close()
cv_ = X.corr().abs().values.copy(); np.fill_diagonal(cv_, 0)
R["pairs_gt09"] = int((np.triu(cv_) > 0.9).sum())

# boxplots of top-6 features by effect size
top6 = mw.feature.head(6).tolist()
fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.6))
for a, c in zip(axes.ravel(), top6):
    sns.boxplot(x=y.map({0: "Benign", 1: "Malignant"}), y=X[c], ax=a, palette=["#4C72B0", "#C44E52"],
                order=["Benign", "Malignant"], fliersize=2)
    a.set_title(c, fontsize=9); a.set_xlabel(""); a.set_ylabel("")
fig.suptitle("Distribution of the six most discriminative features by diagnosis", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "fig_box.png"); plt.close()

# ---------- models ----------
def pipe(est, scale=True):
    steps = [("scale", StandardScaler())] if scale else []
    return Pipeline(steps + [("clf", est)])

models = {
    "Logistic Regression": pipe(LogisticRegression(C=1.0, max_iter=5000)),
    "k-Nearest Neighbors": pipe(KNeighborsClassifier(n_neighbors=7)),
    "SVM (RBF kernel)": pipe(SVC(C=1.0, gamma="scale", probability=True, random_state=SEED)),
    "Decision Tree": pipe(DecisionTreeClassifier(max_depth=5, random_state=SEED), scale=False),
    "Random Forest": pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1), scale=False),
    "Gradient Boosting": pipe(GradientBoostingClassifier(random_state=SEED), scale=False),
    "Gaussian Naive Bayes": pipe(GaussianNB(), scale=False),
    "Multilayer Perceptron": pipe(MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=2000,
                                                random_state=SEED)),
}
abbr = {"Logistic Regression": "LR", "k-Nearest Neighbors": "KNN", "SVM (RBF kernel)": "SVM",
        "Decision Tree": "DT", "Random Forest": "RF", "Gradient Boosting": "GB",
        "Gaussian Naive Bayes": "NB", "Multilayer Perceptron": "MLP"}

# hold-out split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
R["n_train"], R["n_test"] = len(Xtr), len(Xte)
R["test_mal"] = int(yte.sum())

# hyper-parameter tuning on the training set (inner 5-fold CV)
grids = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "k-Nearest Neighbors": {"clf__n_neighbors": [3, 5, 7, 9, 11, 15], "clf__weights": ["uniform", "distance"]},
    "SVM (RBF kernel)": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01, 0.1]},
    "Decision Tree": {"clf__max_depth": [3, 4, 5, 7, None], "clf__min_samples_leaf": [1, 3, 5]},
    "Random Forest": {"clf__max_features": ["sqrt", 0.3, 0.5], "clf__min_samples_leaf": [1, 2, 4]},
    "Gradient Boosting": {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.05, 0.1],
                          "clf__max_depth": [2, 3]},
    "Gaussian Naive Bayes": {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
    "Multilayer Perceptron": {"clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1]},
}
inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
best_params = {}
for name, m in models.items():
    gs = GridSearchCV(m, grids[name], cv=inner, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
    models[name] = gs.best_estimator_
    best_params[name] = {k.replace("clf__", ""): (v if not isinstance(v, float) else float(v))
                         for k, v in gs.best_params_.items()}
R["best_params"] = best_params

# repeated stratified CV on the full dataset with tuned settings: 10 folds x 5 repeats
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED)
folds = list(rskf.split(X, y))
cv = {n: {k: [] for k in ["acc", "prec", "rec", "spec", "f1", "auc", "mcc", "brier"]} for n in models}
t_fit = {n: 0.0 for n in models}
for tr, te in folds:
    for name, m in models.items():
        mm = clone(m); t0 = time.time(); mm.fit(X.iloc[tr], y.iloc[tr]); t_fit[name] += time.time() - t0
        pr = mm.predict_proba(X.iloc[te])[:, 1]; pd_ = (pr >= .5).astype(int); yt = y.iloc[te]
        tn, fp, fn, tp = confusion_matrix(yt, pd_).ravel()
        cv[name]["acc"].append(accuracy_score(yt, pd_)); cv[name]["prec"].append(precision_score(yt, pd_, zero_division=0))
        cv[name]["rec"].append(recall_score(yt, pd_)); cv[name]["spec"].append(tn / (tn + fp))
        cv[name]["f1"].append(f1_score(yt, pd_)); cv[name]["auc"].append(roc_auc_score(yt, pr))
        cv[name]["mcc"].append(matthews_corrcoef(yt, pd_)); cv[name]["brier"].append(brier_score_loss(yt, pr))
rows = []
for n in models:
    r = {"Model": n}
    for k, v in cv[n].items():
        r[k + "_mean"] = np.mean(v); r[k + "_sd"] = np.std(v, ddof=1)
    r["fit_ms"] = t_fit[n] / len(folds) * 1000
    rows.append(r)
cvdf = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
cvdf.to_csv(OUT + "cv_results.csv", index=False)
pd.DataFrame({n: cv[n]["auc"] for n in models}).to_csv(OUT + "cv_auc_folds.csv", index=False)
pd.DataFrame({n: cv[n]["acc"] for n in models}).to_csv(OUT + "cv_acc_folds.csv", index=False)

# ---------- H1: Friedman + Nemenyi on fold-level AUC and accuracy ----------
def friedman(metric):
    M = np.array([cv[n][metric] for n in models]).T  # folds x models
    chi, p = stats.friedmanchisquare(*M.T)
    ranks = np.array([stats.rankdata(-row) for row in M]).mean(0)
    k, N = M.shape[1], M.shape[0]
    q = 3.031  # q_alpha (0.05) for k=8, Demsar (2006) Table 5
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    return chi, p, dict(zip(models, ranks)), cd
for met in ["auc", "acc"]:
    chi, p, ranks, cd = friedman(met)
    R[f"friedman_{met}"] = {"chi2": chi, "p": p, "ranks": ranks, "cd": cd, "df": len(models) - 1}

# CD-style rank plot (AUC)
rk = pd.Series(R["friedman_auc"]["ranks"]).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.barh([abbr[i] for i in rk.index][::-1], rk.values[::-1], color="#55A868")
best = rk.values[0]; cd = R["friedman_auc"]["cd"]
ax.axvline(best + cd, ls="--", color="k", lw=1)
ax.text(best + cd + .05, 0.1, f"best rank + CD ({cd:.2f})", fontsize=8)
ax.set_xlabel("Mean rank across 50 folds (1 = best ROC-AUC)")
ax.set_title("Friedman mean ranks with Nemenyi critical difference")
fig.tight_layout(); fig.savefig(OUT + "fig_ranks.png"); plt.close()

# ---------- corrected resampled t-test (Nadeau & Bengio) best vs others ----------
def corrected_t(a, b, n_tr, n_te):
    d = np.array(a) - np.array(b); J = len(d)
    var = np.var(d, ddof=1)
    if var == 0: return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return t, p
best_model = cvdf.Model.iloc[0]; R["best_model"] = best_model
n_te = len(X) / 10; n_tr = len(X) - n_te
ct = []
for n in models:
    if n == best_model: continue
    t_a, p_a = corrected_t(cv[best_model]["auc"], cv[n]["auc"], n_tr, n_te)
    t_c, p_c = corrected_t(cv[best_model]["acc"], cv[n]["acc"], n_tr, n_te)
    ct.append([n, t_a, p_a, t_c, p_c])
ct = pd.DataFrame(ct, columns=["Model", "t_auc", "p_auc", "t_acc", "p_acc"])
ct["p_auc_holm"] = np.nan
order = ct.p_auc.argsort().values; m_ = len(ct)
adj = np.maximum.accumulate([min(1, ct.p_auc.iloc[i] * (m_ - j)) for j, i in enumerate(order)])
ct.loc[ct.index[order], "p_auc_holm"] = adj
ct.to_csv(OUT + "corrected_t.csv", index=False)

# boxplot of fold AUC
aucdf = pd.DataFrame({abbr[n]: cv[n]["auc"] for n in cvdf.Model})
fig, ax = plt.subplots(figsize=(6.8, 3.4))
sns.boxplot(data=aucdf, ax=ax, color="#8DA0CB", fliersize=2)
ax.set_ylabel("ROC-AUC per fold"); ax.set_title("Spread of ROC-AUC across 50 cross-validation folds")
fig.tight_layout(); fig.savefig(OUT + "fig_cvbox.png"); plt.close()

# ---------- hold-out test evaluation ----------
test_rows, probs = [], {}
for n, m in models.items():
    mm = clone(m).fit(Xtr, ytr); pr = mm.predict_proba(Xte)[:, 1]; pd_ = (pr >= .5).astype(int)
    probs[n] = pr
    tn, fp, fn, tp = confusion_matrix(yte, pd_).ravel()
    test_rows.append({"Model": n, "acc": accuracy_score(yte, pd_), "prec": precision_score(yte, pd_),
                      "rec": recall_score(yte, pd_), "spec": tn / (tn + fp), "f1": f1_score(yte, pd_),
                      "auc": roc_auc_score(yte, pr), "TP": tp, "FN": fn, "FP": fp, "TN": tn})
testdf = pd.DataFrame(test_rows).sort_values("auc", ascending=False)
testdf.to_csv(OUT + "test_results.csv", index=False)

# ROC curves
fig, ax = plt.subplots(figsize=(5.2, 4.6))
for n in testdf.Model:
    f, t, _ = roc_curve(yte, probs[n]); ax.plot(f, t, lw=1.3, label=f"{abbr[n]} (AUC={roc_auc_score(yte, probs[n]):.3f})")
ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("False positive rate (1 - specificity)")
ax.set_ylabel("True positive rate (sensitivity)"); ax.set_title("ROC curves on the hold-out test set")
ax.legend(fontsize=7.5, loc="lower right"); fig.tight_layout(); fig.savefig(OUT + "fig_roc.png"); plt.close()

# confusion matrices for 4 representative
sel = [testdf.Model.iloc[0], "Random Forest", "Decision Tree", "Gaussian Naive Bayes"]
sel = list(dict.fromkeys(sel))[:4]
if len(sel) < 4: sel.append(testdf.Model.iloc[1])
fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
for a, n in zip(axes, sel):
    r = testdf.set_index("Model").loc[n]
    cm = np.array([[r.TN, r.FP], [r.FN, r.TP]]).astype(int)
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=a,
                xticklabels=["Benign", "Malig."], yticklabels=["Benign", "Malig."])
    a.set_title(abbr[n], fontsize=9); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
fig.tight_layout(); fig.savefig(OUT + "fig_cm.png"); plt.close()
R["cm_models"] = sel

# McNemar: best vs each (exact binomial)
best_te = testdf.Model.iloc[0]; R["best_test"] = best_te
pb = (probs[best_te] >= .5).astype(int); mc = []
for n in models:
    if n == best_te: continue
    po = (probs[n] >= .5).astype(int)
    b = int(((pb == yte) & (po != yte)).sum()); c = int(((pb != yte) & (po == yte)).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    mc.append([n, b, c, p])
pd.DataFrame(mc, columns=["Model", "best_right_other_wrong", "best_wrong_other_right", "p_exact"]).to_csv(OUT + "mcnemar.csv", index=False)

# ---------- H2: effect of feature scaling on distance/margin models ----------
sc = []
for n, est in [("k-Nearest Neighbors", KNeighborsClassifier(n_neighbors=7)),
               ("SVM (RBF kernel)", SVC(C=1, gamma="scale", probability=True, random_state=SEED)),
               ("Logistic Regression", LogisticRegression(max_iter=5000)),
               ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]:
    a_s, a_u = [], []
    for tr, te in folds[:50]:
        for scale, store in [(True, a_s), (False, a_u)]:
            mm = pipe(clone(est), scale).fit(X.iloc[tr], y.iloc[tr])
            store.append(accuracy_score(y.iloc[te], mm.predict(X.iloc[te])))
    w = stats.wilcoxon(a_s, a_u) if np.any(np.array(a_s) != np.array(a_u)) else (np.nan, 1.0)
    sc.append([n, np.mean(a_u), np.mean(a_s), w[1]])
scdf = pd.DataFrame(sc, columns=["Model", "acc_unscaled", "acc_scaled", "p_wilcoxon"])
scdf.to_csv(OUT + "scaling.csv", index=False)

# ---------- explainability ----------
best_fit = clone(models[best_model]).fit(Xtr, ytr)
rf_fit = clone(models["Random Forest"]).fit(Xtr, ytr)
pi = permutation_importance(best_fit, Xte, yte, scoring="roc_auc", n_repeats=30, random_state=SEED, n_jobs=-1)
pidf = pd.DataFrame({"feature": X.columns, "mean": pi.importances_mean, "sd": pi.importances_std}).sort_values("mean", ascending=False)
pidf.to_csv(OUT + "perm_best.csv", index=False)
pi_rf = permutation_importance(rf_fit, Xte, yte, scoring="roc_auc", n_repeats=30, random_state=SEED, n_jobs=-1)
pirf = pd.DataFrame({"feature": X.columns, "mean": pi_rf.importances_mean, "sd": pi_rf.importances_std}).sort_values("mean", ascending=False)
pirf.to_csv(OUT + "perm_rf.csv", index=False)
gini = pd.Series(rf_fit.named_steps["clf"].feature_importances_, index=X.columns).sort_values(ascending=False)
gini.to_csv(OUT + "rf_gini.csv")

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
for a, d, t in [(axes[0], gini.head(12)[::-1], "Random Forest: impurity-based importance"),
                (axes[1], pirf.head(12).set_index("feature")["mean"][::-1], "Random Forest: permutation importance")]:
    a.barh(d.index, d.values, color="#DD8452"); a.set_title(t, fontsize=9); a.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(OUT + "fig_imp_rf.png"); plt.close()

# logistic regression coefficients (standardised)
lr_fit = clone(models["Logistic Regression"]).fit(Xtr, ytr)
coef = pd.Series(lr_fit.named_steps["clf"].coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
coef.to_csv(OUT + "lr_coef.csv")
fig, ax = plt.subplots(figsize=(6, 4.4))
cc = coef.head(15)[::-1]
ax.barh(cc.index, cc.values, color=["#C44E52" if v > 0 else "#4C72B0" for v in cc.values])
ax.axvline(0, color="k", lw=.8); ax.tick_params(labelsize=8)
ax.set_xlabel("Standardized coefficient (log-odds of malignancy per 1 SD)")
ax.set_title("Logistic regression: fifteen largest coefficients")
fig.tight_layout(); fig.savefig(OUT + "fig_lrcoef.png"); plt.close()
R["odds_top"] = {k: float(np.exp(v)) for k, v in coef.head(5).items()}

# partial dependence (RF) for top 4 permutation features
top4 = pirf.feature.head(4).tolist()
fig, ax = plt.subplots(figsize=(8, 2.8))
PartialDependenceDisplay.from_estimator(rf_fit, Xtr, top4, ax=ax, n_cols=4, grid_resolution=40)
fig.suptitle("Partial dependence of predicted malignancy probability (Random Forest)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "fig_pdp.png"); plt.close()
R["pdp_feats"] = top4

# ---------- H3: reduced feature subset ----------
# select top-k features by permutation importance computed on training data only, inside each fold
ks = [3, 5, 8, 10, 15, 30]
red = {k: [] for k in ks}
for tr, te in folds[:20]:
    base = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X.iloc[tr], y.iloc[tr])
    order_f = pd.Series(base.feature_importances_, index=X.columns).sort_values(ascending=False).index
    for k in ks:
        cols = order_f[:k]
        mm = clone(models[best_model]).fit(X.iloc[tr][cols], y.iloc[tr])
        red[k].append(roc_auc_score(y.iloc[te], mm.predict_proba(X.iloc[te][cols])[:, 1]))
redf = pd.DataFrame({"k": ks, "auc_mean": [np.mean(red[k]) for k in ks], "auc_sd": [np.std(red[k], ddof=1) for k in ks]})
redf["p_vs_all"] = [stats.wilcoxon(red[k], red[30])[1] if k != 30 else np.nan for k in ks]
redf.to_csv(OUT + "reduced.csv", index=False)
fig, ax = plt.subplots(figsize=(5, 3))
ax.errorbar(redf.k, redf.auc_mean, yerr=redf.auc_sd, marker="o", capsize=3)
ax.set_xlabel("Number of features retained"); ax.set_ylabel("Mean ROC-AUC (20 folds)")
ax.set_title(f"{abbr[best_model]} performance with reduced feature sets"); fig.tight_layout()
fig.savefig(OUT + "fig_reduced.png"); plt.close()

# ---------- threshold analysis on best model (CV out-of-fold) ----------
oof = np.zeros(len(X)); skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
for tr, te in skf.split(X, y):
    oof[te] = clone(models[best_model]).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
th = np.round(np.arange(0.05, 0.96, 0.05), 2); trow = []
for t in th:
    p_ = (oof >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, p_).ravel()
    trow.append([t, tp / (tp + fn), tn / (tn + fp), fn, fp])
thdf = pd.DataFrame(trow, columns=["threshold", "sensitivity", "specificity", "FN", "FP"])
thdf.to_csv(OUT + "threshold.csv", index=False)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(thdf.threshold, thdf.sensitivity, marker=".", label="Sensitivity")
ax.plot(thdf.threshold, thdf.specificity, marker=".", label="Specificity")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Rate"); ax.legend(); ax.set_ylim(.8, 1.005)
ax.set_title(f"Effect of decision threshold ({abbr[best_model]}, out-of-fold)")
fig.tight_layout(); fig.savefig(OUT + "fig_thresh.png"); plt.close()

# calibration
fig, ax = plt.subplots(figsize=(4.8, 4))
oof_all = {}
for n in ["Logistic Regression", "Random Forest", "Gaussian Naive Bayes", "SVM (RBF kernel)"]:
    o = np.zeros(len(X))
    for tr, te in skf.split(X, y):
        o[te] = clone(models[n]).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    oof_all[n] = o
    fp_, mp_ = calibration_curve(y, o, n_bins=8, strategy="quantile")
    ax.plot(mp_, fp_, marker="o", ms=3, label=f"{abbr[n]} (Brier={brier_score_loss(y, o):.3f})")
ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction malignant")
ax.set_title("Reliability (calibration) curves"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT + "fig_calib.png"); plt.close()

# learning curve best
sizes, trs, vas = learning_curve(clone(models[best_model]), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                 train_sizes=np.linspace(.1, 1, 8), scoring="roc_auc", n_jobs=-1)
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(sizes, trs.mean(1), "o-", label="Training"); ax.plot(sizes, vas.mean(1), "o-", label="Validation")
ax.fill_between(sizes, vas.mean(1) - vas.std(1), vas.mean(1) + vas.std(1), alpha=.2)
ax.set_xlabel("Training examples"); ax.set_ylabel("ROC-AUC"); ax.legend(); ax.set_title(f"Learning curve ({abbr[best_model]})")
fig.tight_layout(); fig.savefig(OUT + "fig_lc.png"); plt.close()
R["lc"] = {"sizes": sizes.tolist(), "val": vas.mean(1).tolist(), "train": trs.mean(1).tolist()}

json.dump(R, open(OUT + "summary.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
for f in ["cv_results", "test_results", "corrected_t", "mcnemar", "scaling", "reduced", "threshold"]:
    print("\n==", f); print(pd.read_csv(OUT + f + ".csv").round(4).to_string())
print(mw.round(4).to_string()); print(pidf.head(10).round(4)); print(pirf.head(10).round(4)); print(gini.head(10).round(4)); print(coef.head(10).round(3))

