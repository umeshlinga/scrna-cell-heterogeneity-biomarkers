"""
de_enrichment.py
================
Differential expression + pathway enrichment for CRC TME.
GSE132465: Tumor vs Normal, Stage comparisons, cell-type specific.
Key markers: MDK, COL1A1, TMSB4X (from your resume)
Methods: Wilcoxon, PyDESeq2, GSEApy (KEGG/GO/Reactome/MSigDB)
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
import networkx as nx
from loguru import logger
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
Path("results/figures").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)

try:
    import gseapy as gp
    GSEAPY = True
except ImportError:
    GSEAPY = False

LOGFC=0.5; PVAL=0.05
ENRICHR_LIBS = ["KEGG_2021_Human","GO_Biological_Process_2021",
                "Reactome_2022","MSigDB_Hallmark_2020"]

# Key CRC biomarkers from literature (matches your resume)
CRC_BIOMARKERS = {
    "MDK":     "Tumor progression, poor prognosis",
    "COL1A1":  "CAF activation, fibrosis",
    "TMSB4X":  "Actin dynamics, metastasis",
    "CEACAM5": "Epithelial tumor marker",
    "MKI67":   "Proliferation",
    "TOP2A":   "DNA replication, chemo target",
    "VEGFA":   "Angiogenesis",
    "SPP1":    "Macrophage-tumor crosstalk",
}

def wilcoxon_de(adata, cell_type, g1, g2, cond_col="condition"):
    ct_col = "cell_type" if "cell_type" in adata.obs else "leiden"
    mask = (adata.obs[ct_col].str.replace(" ","_")==cell_type.replace(" ","_")) &            adata.obs[cond_col].isin([g1,g2])
    sub = adata[mask].copy()
    if sub.n_obs < 20: return pd.DataFrame()
    sub.obs["group"] = sub.obs[cond_col]
    sc.tl.rank_genes_groups(sub,"group",groups=[g1],reference=g2,
                            method="wilcoxon",n_genes=sub.n_vars,use_raw=False)
    res = sc.get.rank_genes_groups_df(sub,group=g1,pval_cutoff=1.0)
    res.columns = ["gene","log2FoldChange","pval","padj","pct_1","pct_2"]
    res["cell_type"]  = cell_type
    res["comparison"] = f"{g1}_vs_{g2}"
    res["method"]     = "Wilcoxon"
    res["significant"]= (res["padj"]<PVAL)&(res["log2FoldChange"].abs()>LOGFC)
    return res

def run_all_de(adata):
    logger.info("Running DE (all cell types x comparisons)...")
    cond_col = "condition" if "condition" in adata.obs else None
    ct_col   = "cell_type" if "cell_type" in adata.obs else "leiden"
    comparisons = [("Tumor","Normal")]
    if "stage" in adata.obs:
        comparisons += [("IV","I"),("III","I")]
    results = []
    for ct in adata.obs[ct_col].unique():
        for g1,g2 in comparisons:
            col = cond_col if g1 in ["Tumor","Normal"] else "stage"
            if col not in adata.obs: continue
            conds = adata.obs.loc[
                adata.obs[ct_col].str.replace(" ","_")==ct.replace(" ","_"), col].unique()
            if g1 not in conds or g2 not in conds: continue
            res = wilcoxon_de(adata, ct, g1, g2, col)
            if not res.empty: results.append(res)
    if not results: return pd.DataFrame()
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv("results/tables/de_results_all.csv", index=False)
    # Save biomarker-specific results
    bm_res = combined[combined["gene"].isin(CRC_BIOMARKERS.keys())]
    bm_res.to_csv("results/tables/de_known_biomarkers.csv", index=False)
    logger.success(f"DE complete | {len(combined):,} tests | {combined['significant'].sum():,} sig")
    return combined

def run_enrichment(de_results):
    if not GSEAPY or de_results.empty: return
    logger.info("Running pathway enrichment (Enrichr + GSEA)...")
    all_enrich = []
    for (comp,ct), grp in de_results[de_results["significant"]].groupby(["comparison","cell_type"]):
        genes = grp["gene"].dropna().unique().tolist()
        if len(genes) < 5: continue
        for lib in ENRICHR_LIBS:
            try:
                enr = gp.enrichr(gene_list=genes,gene_sets=lib,
                                 organism="Human",outdir=None,cutoff=PVAL)
                df  = enr.results
                df["comparison"]=comp; df["cell_type"]=ct; df["library"]=lib
                all_enrich.append(df)
            except Exception as e:
                logger.warning(f"  Enrichr {lib} failed: {e}")
    if all_enrich:
        pd.concat(all_enrich).to_csv("results/tables/enrichment_results_all.csv",index=False)
        logger.success("Enrichment results saved.")

def build_ppi_network(de_results, comp="Tumor_vs_Normal", ct="Tumor_Epithelial"):
    import urllib.request, json
    mask = (de_results["comparison"]==comp) &            (de_results["cell_type"].str.replace(" ","_")==ct.replace(" ","_")) &            de_results["significant"]
    genes = de_results[mask].nlargest(50,"log2FoldChange")["gene"].tolist()
    if len(genes) < 5: return
    logger.info(f"Building PPI: {comp} | {ct} | {len(genes)} genes")
    try:
        url = (f"https://string-db.org/api/json/network?"
               f"identifiers={'%0d'.join(genes)}&species=9606"
               f"&required_score=700&caller_identity=crc_scrna_pipeline")
        with urllib.request.urlopen(url,timeout=30) as r:
            edges = json.loads(r.read())
        G = nx.Graph()
        G.add_nodes_from(genes)
        for e in edges:
            G.add_edge(e["preferredName_A"],e["preferredName_B"],weight=e["score"])
        G.remove_nodes_from(list(nx.isolates(G)))
        pr = nx.pagerank(G,weight="weight")
        pd.DataFrame({"gene":list(pr.keys()),"pagerank":list(pr.values())}).sort_values(
            "pagerank",ascending=False).to_csv(
            f"results/tables/ppi_nodes_{comp}_{ct}.csv",index=False)
        logger.success(f"PPI: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        _plot_ppi(G,pr,comp,ct)
    except Exception as e:
        logger.warning(f"PPI failed: {e}")

def _plot_ppi(G,pr,comp,ct):
    fig,ax = plt.subplots(figsize=(12,10))
    pos = nx.spring_layout(G,seed=42,k=2/np.sqrt(max(G.number_of_nodes(),1)))
    sizes  = [pr.get(n,0)*50000+100 for n in G.nodes()]
    bc     = list(nx.betweenness_centrality(G,weight="weight").values())
    nx.draw_networkx_nodes(G,pos,node_size=sizes,node_color=bc,
                            cmap="YlOrRd",alpha=0.85,ax=ax)
    nx.draw_networkx_edges(G,pos,alpha=0.3,edge_color="#95a5a6",ax=ax)
    top_hubs = sorted(pr,key=pr.get,reverse=True)[:15]
    nx.draw_networkx_labels(G,pos,{n:n for n in top_hubs},font_size=7,ax=ax)
    ax.set_title(f"PPI Network: {ct} | {comp}",fontweight="bold",fontsize=12)
    ax.axis("off"); plt.tight_layout()
    plt.savefig(f"results/figures/ppi_{comp}_{ct}.png",dpi=150,bbox_inches="tight")
    plt.close()

def plot_volcano(de_results, comp="Tumor_vs_Normal", ct="Tumor_Epithelial"):
    mask = (de_results["comparison"]==comp) &            (de_results["cell_type"].str.replace(" ","_")==ct.replace(" ","_"))
    df = de_results[mask].dropna(subset=["log2FoldChange","pval"]).copy()
    if df.empty: return
    df["-log10p"] = -np.log10(df["pval"].clip(1e-300))
    df["sig"] = "NS"
    df.loc[(df["log2FoldChange"]>LOGFC)&(df["pval"]<PVAL),"sig"] = "Up"
    df.loc[(df["log2FoldChange"]<-LOGFC)&(df["pval"]<PVAL),"sig"] = "Down"
    fig,ax = plt.subplots(figsize=(9,7))
    for grp,col in [("Up","#e74c3c"),("Down","#3498db"),("NS","#bdc3c7")]:
        s = df[df["sig"]==grp]
        ax.scatter(s["log2FoldChange"],s["-log10p"],c=col,
                   s=12 if grp!="NS" else 5,alpha=0.75,
                   label=f"{grp} (n={len(s):,})",rasterized=True)
    ax.axvline(LOGFC,color="gray",ls="--",lw=1)
    ax.axvline(-LOGFC,color="gray",ls="--",lw=1)
    ax.axhline(-np.log10(PVAL),color="gray",ls="--",lw=1)
    for gene in CRC_BIOMARKERS:
        row = df[df["gene"]==gene]
        if not row.empty:
            r = row.iloc[0]
            ax.scatter(r["log2FoldChange"],r["-log10p"],
                       c="#e74c3c" if r["log2FoldChange"]>0 else "#3498db",
                       s=60,zorder=5,edgecolor="black",lw=0.5)
            ax.annotate(gene,(r["log2FoldChange"],r["-log10p"]),
                        fontsize=8,fontweight="bold",
                        xytext=(r["log2FoldChange"]+0.1,r["-log10p"]+1),
                        arrowprops=dict(arrowstyle="-",color="gray",lw=0.8))
    n_up=(df["sig"]=="Up").sum(); n_dn=(df["sig"]=="Down").sum()
    ax.set_xlabel("log2 Fold Change",fontsize=12)
    ax.set_ylabel("-log10(p-value)",fontsize=12)
    ax.set_title(f"Volcano: {ct} | {comp}\n↑{n_up} up  ↓{n_dn} down",
                 fontsize=12,fontweight="bold")
    ax.legend(frameon=False,fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"results/figures/volcano_{comp}_{ct}.png",dpi=150,bbox_inches="tight")
    plt.close()
    logger.info(f"Volcano saved: {comp} | {ct}")

def run_full_de(adata):
    logger.info("="*60)
    logger.info("Starting DE + Enrichment + PPI Pipeline")
    logger.info("="*60)
    de = run_all_de(adata)
    if not de.empty:
        for comp in ["Tumor_vs_Normal","IV_vs_I"]:
            for ct in ["Tumor_Epithelial","T_cells","Fibroblasts"]:
                plot_volcano(de,comp,ct)
        run_enrichment(de)
        build_ppi_network(de,"Tumor_vs_Normal","Tumor_Epithelial")
    return de

if __name__ == "__main__":
    adata = sc.read_h5ad("data/processed/clustered_annotated.h5ad")
    run_full_de(adata)
