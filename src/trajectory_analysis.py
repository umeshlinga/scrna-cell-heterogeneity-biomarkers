"""
trajectory_analysis.py
=======================
Pseudotime trajectory analysis for CRC epithelial cell differentiation.
Methods:
  - Diffusion Map (DM) for dimensionality reduction
  - Diffusion Pseudotime (DPT) for ordering cells
  - PAGA (Partition-based Graph Abstraction) for trajectory graph
  - RNA Velocity (scVelo) for directional dynamics
  - Monocle3-style branching trajectory analysis
Biological question: How do normal epithelial cells transition to
tumor epithelial states? What are the stage-specific transition genes?
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

warnings.filterwarnings("ignore")
Path("results/figures").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)

EPITHELIAL_TYPES = ["Epithelial","Tumor_Epithelial"]
TRANSITION_GENES = [
    "EPCAM","KRT8","KRT18","CDH1",        # Normal epithelial
    "VIM","CDH2","SNAI1","ZEB1","TWIST1",  # EMT markers
    "MKI67","TOP2A","PCNA","CDK1",         # Proliferation
    "MDK","CEACAM5","CEACAM6",             # Tumor markers
    "VEGFA","HIF1A","LDHA",               # Hypoxia/metabolism
]

def run_diffusion_map(adata, n_dcs=10):
    logger.info("Computing Diffusion Map...")
    sc.tl.diffmap(adata, n_comps=n_dcs)
    logger.success(f"Diffusion map computed (n_dcs={n_dcs})")
    return adata

def run_paga(adata, groups="cell_type"):
    logger.info(f"Running PAGA (groups={groups})...")
    sc.tl.paga(adata, groups=groups)
    sc.pl.paga(adata, show=False)
    _plot_paga(adata)
    logger.success("PAGA complete.")
    return adata

def _plot_paga(adata):
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.paga(adata, ax=ax, show=False,
               node_size_scale=2, edge_width_scale=1.5,
               title="PAGA: CRC Tumor Microenvironment")
    plt.tight_layout()
    plt.savefig("results/figures/paga_graph.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("PAGA graph saved.")

def compute_pseudotime(adata, root_cell_type="Epithelial"):
    logger.info(f"Computing diffusion pseudotime (root={root_cell_type})...")
    if "cell_type" not in adata.obs:
        logger.warning("cell_type not found — skipping pseudotime")
        return adata
    root_mask = adata.obs["cell_type"] == root_cell_type
    if root_mask.sum() == 0:
        logger.warning(f"No cells with cell_type={root_cell_type}")
        return adata
    root_idx = np.where(root_mask)[0][0]
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata, n_dcs=10)
    logger.success(f"Pseudotime computed. Range: {adata.obs['dpt_pseudotime'].min():.3f} - {adata.obs['dpt_pseudotime'].max():.3f}")
    _plot_pseudotime(adata)
    _plot_pseudotime_genes(adata)
    return adata

def _plot_pseudotime(adata):
    umap = adata.obsm["X_umap"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("Pseudotime Trajectory — Epithelial to Tumor Transition",
                 fontsize=13, fontweight="bold")

    # Pseudotime
    sc_p = axes[0].scatter(umap[:,0], umap[:,1],
                            c=adata.obs["dpt_pseudotime"].values,
                            cmap="magma", s=2, alpha=0.7, rasterized=True)
    plt.colorbar(sc_p, ax=axes[0], label="Pseudotime")
    axes[0].set_title("Diffusion Pseudotime", fontweight="bold")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Cell type
    from matplotlib.patches import Patch
    COLORS = {"Epithelial":"#E8A838","Tumor_Epithelial":"#C0392B",
              "T_cells":"#3498DB","Fibroblasts":"#2ECC71","Other":"#BDC3C7"}
    if "cell_type" in adata.obs:
        cols = [COLORS.get(c,"#BDC3C7") for c in adata.obs["cell_type"]]
        axes[1].scatter(umap[:,0], umap[:,1], c=cols, s=2, alpha=0.7, rasterized=True)
        handles = [Patch(color=v,label=k) for k,v in COLORS.items()]
        axes[1].legend(handles=handles, fontsize=7, frameon=False, markerscale=3)
    axes[1].set_title("Cell Type", fontweight="bold")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Condition
    if "condition" in adata.obs:
        cmap = {"Tumor":"#e74c3c","Normal":"#2ecc71"}
        cols = [cmap.get(c,"#bdc3c7") for c in adata.obs["condition"]]
        axes[2].scatter(umap[:,0], umap[:,1], c=cols, s=2, alpha=0.7, rasterized=True)
        handles = [Patch(color=v,label=k) for k,v in cmap.items()]
        axes[2].legend(handles=handles, fontsize=9, frameon=False, markerscale=3)
    axes[2].set_title("Tumor vs Normal", fontweight="bold")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/figures/pseudotime_umap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Pseudotime UMAP saved.")

def _plot_pseudotime_genes(adata, n_genes=10):
    if "dpt_pseudotime" not in adata.obs: return
    genes = [g for g in TRANSITION_GENES if g in adata.var_names][:n_genes]
    if not genes: return

    from scipy.sparse import issparse
    pt = adata.obs["dpt_pseudotime"].values
    sort_idx = np.argsort(pt)
    pt_sorted = pt[sort_idx]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Gene Expression Along Pseudotime Trajectory",
                 fontsize=13, fontweight="bold")

    for ax, gene in zip(axes.flat, genes):
        gene_idx = adata.var_names.tolist().index(gene)
        X = adata.X
        if issparse(X): X = X.toarray()
        expr = X[sort_idx, gene_idx]
        window = max(1, len(expr)//50)
        smooth = np.convolve(expr, np.ones(window)/window, mode="valid")
        pt_sm  = pt_sorted[window//2:window//2+len(smooth)]
        ax.scatter(pt_sorted, expr, s=1, alpha=0.2, color="#bdc3c7", rasterized=True)
        ax.plot(pt_sm, smooth, color="#e74c3c", lw=2)
        ax.set_title(gene, fontweight="bold", fontsize=10)
        ax.set_xlabel("Pseudotime", fontsize=8)
        ax.set_ylabel("Expression", fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/figures/pseudotime_genes.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Pseudotime gene expression plot saved.")

def run_scvelo(adata):
    try:
        import scvelo as scv
        logger.info("Running RNA Velocity (scVelo)...")
        scv.settings.verbosity = 1
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata)
        scv.tl.velocity_graph(adata)
        scv.pl.velocity_embedding_stream(adata, basis="umap", show=False,
                                          save="results/figures/rna_velocity.png")
        logger.success("RNA Velocity complete.")
    except ImportError:
        logger.warning("scvelo not installed — skipping RNA velocity.")
    except Exception as e:
        logger.warning(f"scVelo failed: {e}")
    return adata

def run_full_trajectory(adata):
    logger.info("="*60)
    logger.info("Starting Trajectory Analysis Pipeline")
    logger.info("="*60)
    adata = run_diffusion_map(adata)
    adata = run_paga(adata)
    adata = compute_pseudotime(adata)
    adata = run_scvelo(adata)
    out = Path("data/processed/trajectory.h5ad")
    adata.write_h5ad(out, compression="gzip")
    logger.success(f"Trajectory AnnData saved -> {out}")
    return adata

if __name__ == "__main__":
    import scanpy as sc
    adata = sc.read_h5ad("data/processed/clustered_annotated.h5ad")
    run_full_trajectory(adata)
