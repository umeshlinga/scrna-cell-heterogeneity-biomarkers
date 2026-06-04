"""
ml_biomarker.py
===============
ML pipeline for CRC cell-type heterogeneity biomarker discovery.
GSE132465: Tumor vs Normal, Stage I-IV classification.
Models: XGBoost, LightGBM, Random Forest, SVM, Logistic Regression
Key features: MDK, COL1A1, TMSB4X stage-specific markers (your resume)
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, roc_curve, auc, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
Path("results/figures").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)
Path("results/models").mkdir(parents=True, exist_ok=True)

try: import xgboost as xgb; XGB=True
except ImportError: XGB=False

try: import lightgbm as lgb; LGB=True
except ImportError: LGB=False

try: import shap; SHAP=True
except ImportError: SHAP=False

# Stage-specific biomarkers from Lee et al. 2020 + your resume
STAGE_MARKERS = {
    "MDK":     "Stage-specific, poor prognosis",
    "COL1A1":  "Fibroblast activation by stage",
    "TMSB4X":  "Epithelial dynamics",
    "CEACAM5": "Epithelial tumor marker",
    "MKI67":   "Proliferation index",
    "TOP2A":   "Replication stress",
    "VEGFA":   "Angiogenic switch",
    "SPP1":    "Macrophage polarization",
}

def get_models():
    m = {
        "Random_Forest":       RandomForestClassifier(n_estimators=300,
            class_weight="balanced",random_state=42,n_jobs=-1),
        "Logistic_Regression": LogisticRegression(C=1.0,
            class_weight="balanced",max_iter=1000,random_state=42),
        "SVM":                 SVC(kernel="rbf",probability=True,
            class_weight="balanced",random_state=42),
    }
    if XGB: m["XGBoost"] = xgb.XGBClassifier(n_estimators=300,max_depth=6,
        learning_rate=0.05,use_label_encoder=False,eval_metric="logloss",
        random_state=42,n_jobs=-1)
    if LGB: m["LightGBM"] = lgb.LGBMClassifier(n_estimators=300,
        learning_rate=0.05,class_weight="balanced",
        random_state=42,n_jobs=-1,verbose=-1)
    return m

def build_feature_matrix(adata, de_results, comparison, cell_type, n_feat=300):
    cond_col = "condition" if comparison in ["Tumor_vs_Normal"] else "stage"
    ct_col   = "cell_type" if "cell_type" in adata.obs else "leiden"
    g1,g2    = comparison.split("_vs_")
    if cond_col not in adata.obs: return None,None,None
    mask = (adata.obs[ct_col].str.replace(" ","_")==cell_type.replace(" ","_")) &            adata.obs[cond_col].isin([g1,g2])
    sub = adata[mask]
    if sub.n_obs < 50: return None,None,None
    sig = de_results[(de_results["comparison"]==comparison) &
                     (de_results["cell_type"].str.replace(" ","_")==cell_type.replace(" ","_")) &
                     de_results["significant"]].copy()
    sig["abs_lfc"] = sig["log2FoldChange"].abs()
    genes = sig.nlargest(n_feat,"abs_lfc")["gene"].tolist()
    genes = [g for g in genes if g in sub.var_names]
    if len(genes) < 10:
        genes = [g for g in STAGE_MARKERS if g in sub.var_names]
    if not genes: return None,None,None
    X = sub[:,genes].layers.get("log1p", sub[:,genes].X)
    if issparse(X): X = X.toarray()
    y = LabelEncoder().fit_transform(sub.obs[cond_col].values)
    logger.info(f"  {comparison}|{cell_type}: {X.shape} | classes:{np.bincount(y)}")
    return X.astype(np.float32), y, genes

def nested_cv(X, y, name, model, folds=5):
    scaler = StandardScaler()
    cv     = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_aucs = []
    all_y, all_p = [], []
    for tr,te in cv.split(X,y):
        Xtr=scaler.fit_transform(X[tr]); Xte=scaler.transform(X[te])
        model.fit(Xtr,y[tr])
        p=model.predict_proba(Xte)[:,1]
        fold_aucs.append(roc_auc_score(y[te],p))
        all_y.extend(y[te]); all_p.extend(p)
    mean_auc = np.mean(fold_aucs)
    logger.info(f"    {name}: ROC-AUC={mean_auc:.4f}±{np.std(fold_aucs):.4f}")
    return {"model":name,"roc_auc":round(mean_auc,4),
            "roc_auc_std":round(np.std(fold_aucs),4),
            "y_true":np.array(all_y),"y_prob":np.array(all_p)}

def run_shap(X, y, feat_names, comparison, cell_type):
    if not SHAP: return None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=200,class_weight="balanced",
                                random_state=42,n_jobs=-1)
    rf.fit(Xs,y)
    idx = np.random.choice(len(X),min(400,len(X)),replace=False)
    try:
        exp = shap.TreeExplainer(rf)
        sv  = exp.shap_values(Xs[idx])
        sv  = sv[1] if isinstance(sv,list) else sv
        df  = pd.DataFrame({"gene":feat_names,
                             "mean_shap":np.abs(sv).mean(axis=0)}).sort_values(
                                 "mean_shap",ascending=False)
        df.to_csv(f"results/tables/shap_{comparison}_{cell_type}.csv",index=False)
        # Plot SHAP barplot
        fig,ax = plt.subplots(figsize=(10,7))
        top = df.head(25)
        colors = plt.cm.RdBu_r(np.linspace(0.1,0.9,len(top)))
        ax.barh(top["gene"][::-1],top["mean_shap"][::-1],color=colors)
        ax.set_xlabel("Mean |SHAP Value|"); 
        ax.set_title(f"SHAP: {cell_type} | {comparison}",fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"results/figures/shap_{comparison}_{cell_type}.png",dpi=150,bbox_inches="tight")
        plt.close()
        # Highlight known biomarkers
        known = df[df["gene"].isin(STAGE_MARKERS.keys())]
        if not known.empty:
            logger.info(f"  Known biomarkers in SHAP top features:")
            for _,row in known.head(5).iterrows():
                logger.info(f"    {row['gene']}: {row['mean_shap']:.4f} — {STAGE_MARKERS.get(row['gene'],'')}")
        return df
    except Exception as e:
        logger.warning(f"SHAP failed: {e}"); return None

def plot_roc_curves(X, y, comparison, cell_type):
    models  = get_models()
    scaler  = StandardScaler()
    Xs      = scaler.fit_transform(X)
    cv      = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
    colors = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#f39c12"]
    for (name,model),col in zip(models.items(),colors):
        try:
            yp = cross_val_predict(model,Xs,y,cv=cv,method="predict_proba")[:,1]
            fpr,tpr,_ = roc_curve(y,yp)
            roc_auc   = auc(fpr,tpr)
            ax1.plot(fpr,tpr,color=col,lw=2,label=f"{name} ({roc_auc:.3f})")
        except Exception as e:
            logger.warning(f"  {name} ROC failed: {e}")
    ax1.plot([0,1],[0,1],"k--",lw=1)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title(f"ROC Curves\n{comparison} | {cell_type}",fontweight="bold")
    ax1.legend(loc="lower right",fontsize=9,frameon=False)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Feature importance (RF)
    rf = RandomForestClassifier(n_estimators=200,class_weight="balanced",
                                random_state=42,n_jobs=-1)
    rf.fit(Xs,y)
    fi = pd.Series(rf.feature_importances_,index=range(len(X[0]))).nlargest(20)
    ax2.barh(range(len(fi)),fi.values[::-1],color="#e74c3c",alpha=0.8)
    ax2.set_title("Top 20 Features (RF)",fontweight="bold")
    ax2.set_xlabel("Feature Importance")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"results/figures/roc_{comparison}_{cell_type}.png",dpi=150,bbox_inches="tight")
    plt.close()

def run_full_ml(adata, de_results):
    logger.info("="*60)
    logger.info("Starting ML Biomarker Pipeline — CRC GSE132465")
    logger.info("="*60)
    priority = [
        ("Tumor_vs_Normal","Tumor_Epithelial"),
        ("Tumor_vs_Normal","T_cells"),
        ("Tumor_vs_Normal","Fibroblasts"),
    ]
    all_results = []
    for comp,ct in priority:
        X,y,feats = build_feature_matrix(adata,de_results,comp,ct)
        if X is None: continue
        logger.info(f"\nAnalysis: {comp} | {ct}")
        fold_res = []
        for name,model in get_models().items():
            res = nested_cv(X,y,name,model)
            res["comparison"]=comp; res["cell_type"]=ct
            all_results.append({k:v for k,v in res.items() if k not in ["y_true","y_prob"]})
            fold_res.append(res)
        plot_roc_curves(X,y,comp,ct)
        run_shap(X,y,feats,comp,ct)
    if all_results:
        df = pd.DataFrame(all_results).sort_values("roc_auc",ascending=False)
        df.to_csv("results/tables/all_model_results.csv",index=False)
        logger.success(f"Best: {df.iloc[0]['model']} AUC={df.iloc[0]['roc_auc']}")
        return df
    return pd.DataFrame()

if __name__ == "__main__":
    adata = sc.read_h5ad("data/processed/clustered_annotated.h5ad")
    de    = pd.read_csv("results/tables/de_results_all.csv")
    run_full_ml(adata, de)
