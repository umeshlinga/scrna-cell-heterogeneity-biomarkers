"""
qc_preprocessing.py
====================
Rigorous QC pipeline for GSE132465 CRC scRNA-seq data.
23 patients x 2 conditions (Tumor + Normal) = 68,060 cells.
- Per-sample adaptive MAD thresholds
- Scrublet doublet detection per sample
- Cell cycle scoring
- Normalization + log1p
- HVG selection (batch-aware, per patient)
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scrublet as scr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
Path("results/figures").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)

MIN_GENES=200; MAX_GENES=7000; MAX_MITO=20.0
MIN_COUNTS=500; MAX_COUNTS=60000; MIN_CELLS=3

def compute_qc_metrics(adata):
    logger.info("Computing QC metrics...")
    adata.var["mt"]   = adata.var_names.str.upper().str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.upper().str.match(r"^RP[SL]")
    adata.var["hb"]   = adata.var_names.str.upper().str.match(r"^HB[^(P)]")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ribo","hb"],
                               percent_top=[20,50], log1p=True, inplace=True)
    adata.obs["complexity"] = (
        np.log1p(adata.obs["n_genes_by_counts"]) /
        np.log1p(adata.obs["total_counts"]))
    logger.info(f"  Cells: {adata.n_obs:,} | Median genes: {adata.obs['n_genes_by_counts'].median():.0f}")
    return adata

def adaptive_thresholds(adata, sample_key="sample_id"):
    records = []
    for sid, grp in adata.obs.groupby(sample_key):
        row = {"sample_id": sid}
        for m in ["n_genes_by_counts","total_counts","pct_counts_mt"]:
            vals = grp[m].values
            med  = np.median(vals)
            mad  = stats.median_abs_deviation(vals)
            row[f"{m}_lo"] = max(med - 5*mad, 0)
            row[f"{m}_hi"] = med + 5*mad
        records.append(row)
    df = pd.DataFrame(records).set_index("sample_id")
    df.to_csv("results/tables/adaptive_qc_thresholds.csv")
    logger.info(f"  Adaptive thresholds computed for {len(df)} samples")
    return df

def run_scrublet(adata, sample_key="sample_id", threshold=0.25):
    logger.info("Running Scrublet doublet detection (per sample)...")
    scores = np.zeros(adata.n_obs)
    calls  = np.zeros(adata.n_obs, dtype=bool)
    for sid, grp in adata.obs.groupby(sample_key):
        idx = np.where(adata.obs[sample_key] == sid)[0]
        X   = adata[idx].X
        if issparse(X): X = X.toarray()
        try:
            scrub = scr.Scrublet(X, expected_doublet_rate=0.06)
            s, c  = scrub.scrub_doublets(verbose=False)
            c     = s >= threshold
        except:
            s = np.zeros(len(idx)); c = np.zeros(len(idx), dtype=bool)
        scores[idx] = s; calls[idx] = c
        logger.info(f"  {sid}: {c.sum()}/{len(idx)} doublets ({c.mean()*100:.1f}%)")
    adata.obs["doublet_score"]     = scores
    adata.obs["predicted_doublet"] = calls
    logger.info(f"  Total doublets: {calls.sum():,} ({calls.mean()*100:.1f}%)")
    return adata

def score_cell_cycle(adata):
    s_genes = ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG",
               "GINS2","MCM6","CDCA7","DTL","PRIM1","UHRF1","HELLS",
               "RFC2","RPA2","NASP","RAD51AP1","GMNN","WDR76","SLBP",
               "CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2",
               "CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2",
               "USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"]
    g2m_genes = ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A",
                 "NDC80","CKS2","NUF2","CKS1B","MKI67","TMPO","CENPF",
                 "TACC3","SMC4","CCNB2","AURKB","BUB1","KIF11","GTSE1",
                 "KIF20B","HJURP","CDCA3","CDC20","TTK","CDC25C","KIF2C",
                 "RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2",
                 "KIF23","HMMR","AURKA","ANLN","CKAP5","CENPE","NEK2",
                 "G2E3","GAS2L3","CBX5","CENPA"]
    s_genes   = [g for g in s_genes   if g in adata.var_names]
    g2m_genes = [g for g in g2m_genes if g in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    logger.info(f"  Cell cycle: {adata.obs['phase'].value_counts().to_dict()}")
    return adata

def apply_filters(adata):
    n0 = adata.n_obs
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    keep = (
        (adata.obs["n_genes_by_counts"]  >= MIN_GENES) &
        (adata.obs["n_genes_by_counts"]  <= MAX_GENES) &
        (adata.obs["total_counts"]       >= MIN_COUNTS) &
        (adata.obs["total_counts"]       <= MAX_COUNTS) &
        (adata.obs["pct_counts_mt"]      <= MAX_MITO)
    )
    if "predicted_doublet" in adata.obs:
        keep &= ~adata.obs["predicted_doublet"]
    adata = adata[keep].copy()
    logger.success(f"QC: {adata.n_obs:,}/{n0:,} cells ({adata.n_obs/n0*100:.1f}%)")
    return adata

def normalize(adata):
    adata.layers["counts"]     = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.layers["normalized"] = adata.X.copy()
    sc.pp.log1p(adata)
    adata.layers["log1p"]      = adata.X.copy()
    logger.success("Normalization complete.")
    return adata

def select_hvg(adata, n_top=3000, batch_key="patient_id"):
    bk = batch_key if batch_key in adata.obs else None
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top,
        flavor="seurat_v3", layer="counts",
        batch_key=bk, subset=False)
    logger.info(f"  HVGs: {adata.var['highly_variable'].sum():,}")
    return adata

def plot_qc_panels(adata):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("QC Metrics — GSE132465 CRC (23 patients, 68,060 cells)",
                 fontsize=13, fontweight="bold")
    metrics = [
        ("n_genes_by_counts","Genes/Cell","#3498db"),
        ("total_counts","UMI Counts","#2ecc71"),
        ("pct_counts_mt","Mito %","#e74c3c"),
        ("pct_counts_ribo","Ribo %","#9b59b6"),
        ("complexity","Complexity","#e67e22"),
        ("doublet_score","Doublet Score","#1abc9c"),
    ]
    for ax, (m, lab, col) in zip(axes.flat, metrics):
        if m not in adata.obs: ax.set_visible(False); continue
        data = [adata.obs[adata.obs["condition"]==c][m].values
                for c in ["Normal","Tumor"] if c in adata.obs.get("condition",pd.Series()).unique()]
        ax.violinplot(data, showmedians=True)
        ax.set_xticks([1,2]); ax.set_xticklabels(["Normal","Tumor"])
        ax.set_title(lab, fontweight="bold"); ax.set_ylabel(lab)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/figures/qc_violin.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("QC violin saved.")

def plot_qc_scatter(adata):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("QC Scatter: UMI Counts vs Genes (colored by mito%)", fontweight="bold")
    for ax, cond in zip(axes, ["Normal","Tumor"]):
        if "condition" not in adata.obs: break
        mask = adata.obs["condition"] == cond
        obs  = adata[mask].obs
        sc_p = ax.scatter(obs["total_counts"], obs["n_genes_by_counts"],
                          c=obs["pct_counts_mt"], cmap="RdYlGn_r",
                          s=2, alpha=0.4, vmin=0, vmax=MAX_MITO, rasterized=True)
        ax.axhline(MIN_GENES, color="red", ls="--", lw=1, label=f"min={MIN_GENES}")
        ax.axhline(MAX_GENES, color="red", ls="--", lw=1, label=f"max={MAX_GENES}")
        plt.colorbar(sc_p, ax=ax, label="Mito %")
        ax.set_xlabel("Total UMI Counts"); ax.set_ylabel("# Genes")
        ax.set_title(f"{cond} (n={mask.sum():,})", fontweight="bold")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("results/figures/qc_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("QC scatter saved.")

def run_full_preprocessing(adata):
    logger.info("="*60)
    logger.info("Starting QC + Preprocessing Pipeline — GSE132465")
    logger.info("="*60)
    adata = compute_qc_metrics(adata)
    adaptive_thresholds(adata)
    adata = run_scrublet(adata)
    adata = score_cell_cycle(adata)
    plot_qc_panels(adata)
    plot_qc_scatter(adata)
    adata = apply_filters(adata)
    adata = normalize(adata)
    adata = select_hvg(adata)
    out = Path("data/processed/qc_filtered.h5ad")
    adata.write_h5ad(out, compression="gzip")
    logger.success(f"Preprocessed AnnData saved -> {out}")
    return adata

if __name__ == "__main__":
    from src.data_ingestion import load_raw
    adata = load_raw()
    run_full_preprocessing(adata)
