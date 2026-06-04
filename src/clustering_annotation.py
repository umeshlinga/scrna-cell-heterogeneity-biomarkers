"""
clustering_annotation.py
=========================
PCA, Harmony batch correction (per patient), UMAP, Leiden clustering,
and comprehensive cell type annotation for CRC tumor microenvironment.
GSE132465: 23 patients, Tumor + Normal, 9 cell types.
Marker genes from Lee et al. Nature Genetics 2020.
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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

CELL_COLORS = {
    "Epithelial":       "#E8A838",
    "Tumor_Epithelial": "#C0392B",
    "T_cells":          "#3498DB",
    "B_cells":          "#16A085",
    "Myeloid":          "#8E44AD",
    "NK_cells":         "#E74C3C",
    "Fibroblasts":      "#2ECC71",
    "Endothelial":      "#F39C12",
    "Mast_cells":       "#95A5A6",
    "Unknown":          "#BDC3C7",
}

CONDITION_COLORS = {"Tumor":"#e74c3c","Normal":"#2ecc71"}
STAGE_COLORS = {"I":"#3498db","II":"#2ecc71","III":"#f39c12","IV":"#e74c3c"}

CRC_MARKERS = {
    "Epithelial":       ["EPCAM","KRT8","KRT18","KRT19","CDH1","MUC2"],
    "Tumor_Epithelial": ["EPCAM","MKI67","TOP2A","PCNA","MDK","CEACAM5"],
    "T_cells":          ["CD3D","CD3E","CD8A","CD4","FOXP3","GZMB"],
    "B_cells":          ["CD79A","MS4A1","CD19","IGKC","IGHM","BANK1"],
    "Myeloid":          ["CD68","LYZ","S100A8","CD14","FCGR3A","MARCO"],
    "NK_cells":         ["NKG7","GNLY","KLRD1","NCAM1","PRF1","FGFBP2"],
    "Fibroblasts":      ["COL1A1","COL3A1","ACTA2","FAP","PDGFRB","THY1"],
    "Endothelial":      ["PECAM1","VWF","CDH5","MCAM","CLDN5","RAMP2"],
    "Mast_cells":       ["TPSAB1","TPSB2","CPA3","KIT","HPGDS","HDC"],
}

def run_pca(adata, n_comps=50):
    logger.info("Running PCA...")
    sc.tl.pca(adata, n_comps=n_comps, use_highly_variable=True, svd_solver="arpack")
    var = adata.uns["pca"]["variance_ratio"]
    cum = np.cumsum(var)
    n90 = np.searchsorted(cum, 0.90) + 1
    logger.info(f"  PC1:{var[0]*100:.1f}% PC2:{var[1]*100:.1f}% | PCs for 90% var: {n90}")
    _plot_pca_variance(var)
    return adata

def _plot_pca_variance(var):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    pcs = np.arange(1,len(var)+1)
    ax1.plot(pcs, var*100, "o-", color="#3498db", ms=3, lw=1.5)
    ax1.axvline(20, color="red", ls="--", lw=1, label="PC=20")
    ax1.set_xlabel("PC"); ax1.set_ylabel("Variance (%)"); ax1.set_title("Elbow Plot",fontweight="bold")
    ax1.legend(); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax2.plot(pcs, np.cumsum(var)*100, "-", color="#e74c3c", lw=2)
    ax2.axhline(90, color="gray", ls="--", lw=1, label="90%")
    ax2.set_xlabel("# PCs"); ax2.set_ylabel("Cumulative Var (%)"); ax2.set_title("Cumulative Variance",fontweight="bold")
    ax2.legend(); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/figures/pca_variance.png", dpi=150, bbox_inches="tight"); plt.close()

def run_harmony(adata, batch_key="patient_id", n_pcs=30):
    try:
        import harmonypy as hm
        logger.info(f"Harmony batch correction (batch={batch_key})...")
        pca = adata.obsm["X_pca"][:,:n_pcs]
        meta = adata.obs[[batch_key]]
        ho = hm.run_harmony(pca, meta, batch_key, max_iter_harmony=30, random_state=42)
        adata.obsm["X_pca_harmony"] = ho.Z_corr.T
        logger.success("Harmony complete.")
    except ImportError:
        logger.warning("harmonypy unavailable.")
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"][:,:n_pcs]
    return adata

def compute_umap(adata, n_pcs=30):
    rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    logger.info(f"Computing UMAP (rep={rep})...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs, use_rep=rep, random_state=42)
    sc.tl.umap(adata, min_dist=0.3, random_state=42)
    adata.obsm["X_umap_2d"] = adata.obsm["X_umap"].copy()
    sc.tl.umap(adata, min_dist=0.3, n_components=3, random_state=42)
    adata.obsm["X_umap_3d"] = adata.obsm["X_umap"].copy()
    adata.obsm["X_umap"]    = adata.obsm["X_umap_2d"]
    logger.success("UMAP 2D+3D complete.")
    return adata

def leiden_sweep(adata, resolutions=[0.2,0.4,0.6,0.8,1.0,1.2]):
    logger.info("Leiden resolution sweep...")
    records = []
    for res in resolutions:
        key = f"leiden_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=42)
        n = adata.obs[key].nunique()
        lenc = LabelEncoder().fit_transform(adata.obs[key])
        sil  = silhouette_score(adata.obsm["X_umap"], lenc, sample_size=3000)
        records.append({"resolution":res,"n_clusters":n,"silhouette":round(sil,4)})
        logger.info(f"  res={res}: {n} clusters | silhouette={sil:.4f}")
    df = pd.DataFrame(records)
    df.to_csv("results/tables/leiden_sweep.csv", index=False)
    best_res = df.loc[df["silhouette"].idxmax(),"resolution"]
    adata.obs["leiden"] = adata.obs[f"leiden_{best_res}"]
    adata.uns["leiden_resolution"] = best_res
    logger.success(f"Best resolution: {best_res} ({df.loc[df.resolution==best_res,'n_clusters'].values[0]} clusters)")
    _plot_leiden_sweep(df)
    return adata, df

def _plot_leiden_sweep(df):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    ax1.plot(df["resolution"],df["n_clusters"],"s-",color="#3498db",ms=8)
    ax1.set_xlabel("Resolution"); ax1.set_ylabel("# Clusters")
    ax1.set_title("Clusters vs Resolution",fontweight="bold")
    ax2.plot(df["resolution"],df["silhouette"],"o-",color="#e74c3c",ms=8)
    best = df.loc[df["silhouette"].idxmax()]
    ax2.axvline(best["resolution"],color="gray",ls="--",lw=1.5,label=f"Best:{best['resolution']}")
    ax2.set_xlabel("Resolution"); ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Clustering Quality",fontweight="bold"); ax2.legend()
    for ax in [ax1,ax2]:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/figures/leiden_sweep.png",dpi=150,bbox_inches="tight"); plt.close()

def annotate_clusters(adata):
    logger.info("Annotating cell types with CRC marker genes...")
    for ct, markers in CRC_MARKERS.items():
        present = [g for g in markers if g in adata.var_names]
        if len(present) >= 2:
            sc.tl.score_genes(adata, present, score_name=f"score_{ct}")
    score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    annot = {}
    for cl in adata.obs["leiden"].unique():
        mask = adata.obs["leiden"] == cl
        if score_cols:
            best = adata[mask].obs[score_cols].mean().idxmax()
            annot[cl] = best.replace("score_","")
        else:
            annot[cl] = "Unknown"
        logger.info(f"  Cluster {cl} -> {annot[cl]}")
    adata.obs["cell_type"] = adata.obs["leiden"].map(annot)
    pd.DataFrame({"cluster":list(annot.keys()),
                  "cell_type":list(annot.values())}).to_csv(
        "results/tables/cluster_annotation.csv", index=False)
    logger.success(f"Annotated: {adata.obs['cell_type'].value_counts().to_dict()}")
    return adata

def plot_umap_panel(adata):
    umap = adata.obsm["X_umap"]
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.suptitle("UMAP — GSE132465 CRC Tumor Microenvironment
"
                 "(23 patients · 68,060 cells · 9 cell types)",
                 fontsize=14, fontweight="bold")
    obs = adata.obs

    def _scatter(ax, col, title, cmap_d=None, continuous=False):
        if col not in obs.columns: ax.set_visible(False); return
        if continuous:
            sc_p = ax.scatter(umap[:,0],umap[:,1],c=obs[col].values,
                              cmap="viridis",s=1.5,alpha=0.6,rasterized=True)
            plt.colorbar(sc_p,ax=ax,shrink=0.8)
        else:
            cats = obs[col].unique()
            if cmap_d is None:
                pal = plt.cm.get_cmap("tab20",len(cats))
                cmap_d = {c:pal(i) for i,c in enumerate(cats)}
            cols = [cmap_d.get(str(c),"#bdc3c7") for c in obs[col]]
            ax.scatter(umap[:,0],umap[:,1],c=cols,s=1.5,alpha=0.6,rasterized=True)
            import matplotlib.patches as mp
            handles = [mp.Patch(color=cmap_d.get(str(c),"#bdc3c7"),label=str(c)) for c in cats]
            ax.legend(handles=handles,fontsize=6,markerscale=3,
                      loc="lower right",frameon=False,ncol=2)
        ax.set_title(title,fontweight="bold",fontsize=11)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)

    _scatter(axes[0,0],"cell_type",   "Cell Type",      CELL_COLORS)
    _scatter(axes[0,1],"condition",   "Tumor vs Normal",CONDITION_COLORS)
    _scatter(axes[0,2],"stage",       "Cancer Stage",   STAGE_COLORS)
    _scatter(axes[1,0],"patient_id",  "Patient ID")
    _scatter(axes[1,1],"phase",       "Cell Cycle Phase")
    _scatter(axes[1,2],"n_genes_by_counts","# Genes/Cell",continuous=True)

    plt.tight_layout()
    plt.savefig("results/figures/umap_panel.png",dpi=150,bbox_inches="tight")
    plt.close(); logger.info("UMAP panel saved.")

def plot_cell_composition(adata):
    if "cell_type" not in adata.obs or "condition" not in adata.obs: return
    comp = (adata.obs.groupby(["condition","cell_type"]).size()
            .unstack(fill_value=0))
    comp_pct = comp.div(comp.sum(axis=1),axis=0)*100
    colors = [CELL_COLORS.get(ct,"#bdc3c7") for ct in comp_pct.columns]
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
    fig.suptitle("Cell Type Composition: Tumor vs Normal",fontsize=13,fontweight="bold")
    comp_pct.T.plot(kind="bar",ax=ax1,color=["#2ecc71","#e74c3c"])
    ax1.set_xlabel("Cell Type"); ax1.set_ylabel("% of Cells")
    ax1.set_title("% by Condition",fontweight="bold")
    ax1.tick_params(axis="x",rotation=45); ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    comp_pct.plot(kind="bar",stacked=True,ax=ax2,color=colors,edgecolor="white",lw=0.5)
    ax2.set_xlabel("Condition"); ax2.set_ylabel("% of Cells")
    ax2.set_title("Stacked Composition",fontweight="bold")
    ax2.tick_params(axis="x",rotation=0)
    ax2.legend(title="Cell Type",fontsize=8,bbox_to_anchor=(1.05,1),frameon=False)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/figures/cell_composition.png",dpi=150,bbox_inches="tight")
    plt.close(); logger.info("Cell composition saved.")

def run_full_clustering(adata):
    logger.info("="*60)
    logger.info("Starting Clustering + Annotation Pipeline")
    logger.info("="*60)
    adata = run_pca(adata)
    adata = run_harmony(adata)
    adata = compute_umap(adata)
    adata, _ = leiden_sweep(adata)
    adata = annotate_clusters(adata)
    plot_umap_panel(adata)
    plot_cell_composition(adata)
    out = Path("data/processed/clustered_annotated.h5ad")
    adata.write_h5ad(out, compression="gzip")
    logger.success(f"Saved -> {out}")
    return adata

if __name__ == "__main__":
    adata = sc.read_h5ad("data/processed/qc_filtered.h5ad")
    run_full_clustering(adata)
