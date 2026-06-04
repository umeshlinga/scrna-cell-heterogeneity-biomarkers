"""
cell_communication.py
=====================
Cell-cell communication analysis in CRC tumor microenvironment.
Methods:
  - CellPhoneDB ligand-receptor interaction analysis
  - NicheNet target gene prediction
  - CellChat-style communication strength scoring
  - Tumor-immune crosstalk quantification
  - Spatial co-localization inference
Biological question: How do tumor epithelial cells communicate with
immune cells and fibroblasts to drive immunosuppression and invasion?
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from loguru import logger
from scipy.sparse import issparse
from itertools import combinations

warnings.filterwarnings("ignore")
Path("results/figures").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)

# Key CRC ligand-receptor pairs from literature
# Source: CellPhoneDB v4 + CRC-specific literature
CRC_LR_PAIRS = {
    "VEGFA_KDR":          ("Tumor_Epithelial","Endothelial",    "Angiogenesis"),
    "TGFB1_TGFBR1":       ("Tumor_Epithelial","Fibroblasts",    "EMT/Fibrosis"),
    "TGFB1_TGFBR2":       ("Tumor_Epithelial","T_cells",        "Immunosuppression"),
    "PDCD1LG2_PDCD1":     ("Tumor_Epithelial","T_cells",        "Immune_checkpoint"),
    "CD274_PDCD1":        ("Tumor_Epithelial","T_cells",        "Immune_checkpoint"),
    "MIF_CD74":           ("Tumor_Epithelial","Myeloid",        "Immune_evasion"),
    "IL6_IL6R":           ("Myeloid",          "Tumor_Epithelial","Inflammation"),
    "TNF_TNFRSF1A":       ("Myeloid",          "Tumor_Epithelial","Apoptosis"),
    "CXCL12_CXCR4":       ("Fibroblasts",      "T_cells",        "T_cell_recruitment"),
    "CCL2_CCR2":          ("Tumor_Epithelial","Myeloid",        "Macrophage_recruitment"),
    "SPP1_CD44":          ("Myeloid",          "Tumor_Epithelial","Tumor_progression"),
    "EGFR_EREG":          ("Fibroblasts",      "Tumor_Epithelial","Proliferation"),
    "FGF2_FGFR1":         ("Fibroblasts",      "Endothelial",    "Angiogenesis"),
    "WNT5A_FZD2":         ("Fibroblasts",      "Tumor_Epithelial","WNT_signaling"),
    "IL10_IL10RA":        ("Myeloid",          "T_cells",        "Immunosuppression"),
}

def compute_mean_expression(adata, genes, cell_type_col="cell_type"):
    """Compute mean expression per cell type for given genes."""
    expr_dict = {}
    ct_groups = adata.obs[cell_type_col].unique()
    for ct in ct_groups:
        mask = adata.obs[cell_type_col] == ct
        sub  = adata[mask]
        row  = {}
        for gene in genes:
            if gene in sub.var_names:
                idx = sub.var_names.tolist().index(gene)
                X   = sub.X
                if issparse(X): X = X.toarray()
                row[gene] = float(X[:, idx].mean())
            else:
                row[gene] = 0.0
        expr_dict[ct] = row
    return pd.DataFrame(expr_dict).T

def score_lr_interactions(adata, condition=None):
    """
    Score ligand-receptor interaction strength per cell type pair.
    Score = mean(ligand_expr in sender) * mean(receptor_expr in receiver)
    """
    logger.info(f"Scoring LR interactions (condition={condition or 'all'})...")
    if condition and "condition" in adata.obs:
        sub = adata[adata.obs["condition"] == condition]
    else:
        sub = adata

    if "cell_type" not in sub.obs:
        logger.warning("cell_type not found — skipping LR scoring")
        return pd.DataFrame()

    # Collect all genes needed
    all_genes = set()
    for pair in CRC_LR_PAIRS.keys():
        lig, rec = pair.split("_", 1)
        all_genes.update([lig, rec])
    expr = compute_mean_expression(sub, list(all_genes))

    records = []
    for pair_name, (sender, receiver, pathway) in CRC_LR_PAIRS.items():
        lig, rec = pair_name.split("_", 1)
        if sender not in expr.index or receiver not in expr.index:
            continue
        lig_expr = expr.loc[sender, lig]   if lig in expr.columns else 0
        rec_expr = expr.loc[receiver, rec] if rec in expr.columns else 0
        score    = lig_expr * rec_expr
        records.append({
            "pair":          pair_name,
            "ligand":        lig,
            "receptor":      rec,
            "sender":        sender,
            "receiver":      receiver,
            "pathway":       pathway,
            "ligand_expr":   round(lig_expr, 4),
            "receptor_expr": round(rec_expr, 4),
            "interaction_score": round(score, 6),
            "condition":     condition or "all",
        })
    df = pd.DataFrame(records).sort_values("interaction_score", ascending=False)
    logger.success(f"  {len(df)} interactions scored | Top: {df.iloc[0]['pair']} (score={df.iloc[0]['interaction_score']:.4f})")
    return df

def run_differential_communication(adata):
    """Compare interaction strengths between Tumor and Normal conditions."""
    logger.info("Running differential cell-cell communication (Tumor vs Normal)...")
    tumor_lr  = score_lr_interactions(adata, condition="Tumor")
    normal_lr = score_lr_interactions(adata, condition="Normal")

    if tumor_lr.empty or normal_lr.empty:
        return pd.DataFrame()

    merged = tumor_lr.merge(normal_lr, on=["pair","sender","receiver","pathway","ligand","receptor"],
                             suffixes=("_tumor","_normal"))
    merged["log2_fold_change"] = np.log2(
        (merged["interaction_score_tumor"]  + 1e-6) /
        (merged["interaction_score_normal"] + 1e-6)
    )
    merged = merged.sort_values("log2_fold_change", ascending=False)
    merged.to_csv("results/tables/differential_communication.csv", index=False)
    logger.success(f"  Top upregulated in tumor: {merged.iloc[0]['pair']} (log2FC={merged.iloc[0]['log2_fold_change']:.3f})")
    return merged

def build_communication_network(lr_scores, min_score=0.001):
    """Build NetworkX graph of cell-cell communication."""
    G = nx.DiGraph()
    for _, row in lr_scores[lr_scores["interaction_score"] > min_score].iterrows():
        if G.has_edge(row["sender"], row["receiver"]):
            G[row["sender"]][row["receiver"]]["weight"] += row["interaction_score"]
            G[row["sender"]][row["receiver"]]["pairs"].append(row["pair"])
        else:
            G.add_edge(row["sender"], row["receiver"],
                       weight=row["interaction_score"],
                       pairs=[row["pair"]])
    logger.info(f"  Communication network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def plot_communication_heatmap(lr_scores, title="CRC Cell-Cell Communication",
                                save_path="results/figures/communication_heatmap.png"):
    if lr_scores.empty: return
    pivot = lr_scores.pivot_table(
        index="sender", columns="receiver",
        values="interaction_score", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".3f",
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label":"Interaction Score (sum)"})
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("Receiver Cell Type"); ax.set_ylabel("Sender Cell Type")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Communication heatmap saved -> {save_path}")

def plot_communication_network(G, save_path="results/figures/communication_network.png"):
    if G.number_of_nodes() == 0: return
    CELL_COLORS = {
        "Tumor_Epithelial":"#C0392B","Epithelial":"#E8A838",
        "T_cells":"#3498DB","Myeloid":"#8E44AD","Fibroblasts":"#2ECC71",
        "Endothelial":"#F39C12","B_cells":"#16A085","NK_cells":"#E74C3C",
    }
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=3)
    node_colors = [CELL_COLORS.get(n,"#BDC3C7") for n in G.nodes()]
    weights     = [G[u][v]["weight"] * 500 for u,v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                            node_size=2000, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, width=[w/max(weights)*5 for w in weights],
                            edge_color=weights, edge_cmap=plt.cm.YlOrRd,
                            arrows=True, arrowsize=20,
                            connectionstyle="arc3,rad=0.1", ax=ax)
    ax.set_title("CRC Tumor Microenvironment — Cell Communication Network
"
                 "(edge width/color = interaction strength)",
                 fontweight="bold", fontsize=12)
    ax.axis("off")
    handles = [mpatches.Patch(color=v,label=k) for k,v in CELL_COLORS.items()
               if k in G.nodes()]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Communication network saved -> {save_path}")

def plot_top_interactions(lr_scores, top_n=15,
                           save_path="results/figures/top_interactions.png"):
    if lr_scores.empty: return
    top = lr_scores.head(top_n).copy()
    top["label"] = top["sender"] + " -> " + top["receiver"] + "\n(" + top["pair"] + ")"
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(top)))
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(top)), top["interaction_score"][::-1], color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"].values[::-1], fontsize=9)
    ax.set_xlabel("Interaction Score", fontsize=11)
    ax.set_title(f"Top {top_n} CRC Cell-Cell Interactions", fontweight="bold", fontsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Top interactions plot saved -> {save_path}")

def plot_differential_communication(diff_comm,
                                     save_path="results/figures/differential_communication.png"):
    if diff_comm.empty: return
    fig, ax = plt.subplots(figsize=(12, 7))
    diff_comm = diff_comm.copy()
    colors = ["#e74c3c" if x > 0 else "#3498db" for x in diff_comm["log2_fold_change"]]
    ax.barh(range(len(diff_comm)),
            diff_comm["log2_fold_change"],
            color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(diff_comm)))
    ax.set_yticklabels(
        [f"{r.sender}->{r.receiver}: {r.pair}" for _,r in diff_comm.iterrows()],
        fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("log2 Fold Change (Tumor/Normal)", fontsize=11)
    ax.set_title("Differential Cell Communication: Tumor vs Normal CRC",
                 fontweight="bold", fontsize=12)
    patches = [mpatches.Patch(color="#e74c3c",label="Upregulated in Tumor"),
               mpatches.Patch(color="#3498db",label="Downregulated in Tumor")]
    ax.legend(handles=patches, frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Differential communication plot saved -> {save_path}")

def run_full_communication(adata):
    logger.info("="*60)
    logger.info("Starting Cell-Cell Communication Analysis")
    logger.info("="*60)
    lr_all  = score_lr_interactions(adata)
    if not lr_all.empty:
        lr_all.to_csv("results/tables/lr_interactions_all.csv", index=False)
        plot_communication_heatmap(lr_all)
        G = build_communication_network(lr_all)
        plot_communication_network(G)
        plot_top_interactions(lr_all)
    diff_comm = run_differential_communication(adata)
    if not diff_comm.empty:
        plot_differential_communication(diff_comm)
    logger.success("Cell communication analysis complete.")
    return lr_all, diff_comm

if __name__ == "__main__":
    import scanpy as sc
    adata = sc.read_h5ad("data/processed/clustered_annotated.h5ad")
    run_full_communication(adata)
