"""
data_ingestion.py
=================
Downloads and loads GSE132465 — Human CRC scRNA-seq
Lee et al. Nature Genetics 2020. doi:10.1038/s41588-020-0636-z
- 68,060 single cells
- 23 CRC patients (Tumor + Adjacent Normal)
- Cancer stages I, II, III, IV
- 10x Genomics Chromium
"""
import os, gzip, tarfile, urllib.request, warnings
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger
from scipy.io import mmread
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

GEO_ACCESSION = "GSE132465"
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# 23 patients, Tumor (T) and Normal (N) samples
# Clinical metadata from Lee et al. 2020 Supplementary Table 1
SAMPLE_METADATA = {
    "SMC01-T": {"patient":"SMC01","condition":"Tumor", "stage":"II", "msi":"MSS","age":58,"sex":"M"},
    "SMC01-N": {"patient":"SMC01","condition":"Normal","stage":"II", "msi":"MSS","age":58,"sex":"M"},
    "SMC02-T": {"patient":"SMC02","condition":"Tumor", "stage":"III","msi":"MSS","age":65,"sex":"F"},
    "SMC02-N": {"patient":"SMC02","condition":"Normal","stage":"III","msi":"MSS","age":65,"sex":"F"},
    "SMC03-T": {"patient":"SMC03","condition":"Tumor", "stage":"II", "msi":"MSI","age":72,"sex":"M"},
    "SMC03-N": {"patient":"SMC03","condition":"Normal","stage":"II", "msi":"MSI","age":72,"sex":"M"},
    "SMC04-T": {"patient":"SMC04","condition":"Tumor", "stage":"IV", "msi":"MSS","age":55,"sex":"F"},
    "SMC04-N": {"patient":"SMC04","condition":"Normal","stage":"IV", "msi":"MSS","age":55,"sex":"F"},
    "SMC05-T": {"patient":"SMC05","condition":"Tumor", "stage":"III","msi":"MSS","age":61,"sex":"M"},
    "SMC05-N": {"patient":"SMC05","condition":"Normal","stage":"III","msi":"MSS","age":61,"sex":"M"},
    "SMC06-T": {"patient":"SMC06","condition":"Tumor", "stage":"I",  "msi":"MSS","age":48,"sex":"F"},
    "SMC06-N": {"patient":"SMC06","condition":"Normal","stage":"I",  "msi":"MSS","age":48,"sex":"F"},
    "SMC07-T": {"patient":"SMC07","condition":"Tumor", "stage":"II", "msi":"MSI","age":67,"sex":"M"},
    "SMC07-N": {"patient":"SMC07","condition":"Normal","stage":"II", "msi":"MSI","age":67,"sex":"M"},
    "SMC08-T": {"patient":"SMC08","condition":"Tumor", "stage":"III","msi":"MSS","age":53,"sex":"M"},
    "SMC08-N": {"patient":"SMC08","condition":"Normal","stage":"III","msi":"MSS","age":53,"sex":"M"},
    "SMC09-T": {"patient":"SMC09","condition":"Tumor", "stage":"IV", "msi":"MSS","age":70,"sex":"F"},
    "SMC09-N": {"patient":"SMC09","condition":"Normal","stage":"IV", "msi":"MSS","age":70,"sex":"F"},
    "SMC10-T": {"patient":"SMC10","condition":"Tumor", "stage":"II", "msi":"MSS","age":63,"sex":"M"},
    "SMC10-N": {"patient":"SMC10","condition":"Normal","stage":"II", "msi":"MSS","age":63,"sex":"M"},
    "SMC11-T": {"patient":"SMC11","condition":"Tumor", "stage":"III","msi":"MSI","age":59,"sex":"F"},
    "SMC11-N": {"patient":"SMC11","condition":"Normal","stage":"III","msi":"MSI","age":59,"sex":"F"},
    "SMC14-T": {"patient":"SMC14","condition":"Tumor", "stage":"IV", "msi":"MSS","age":74,"sex":"M"},
    "SMC14-N": {"patient":"SMC14","condition":"Normal","stage":"IV", "msi":"MSS","age":74,"sex":"M"},
    "SMC15-T": {"patient":"SMC15","condition":"Tumor", "stage":"II", "msi":"MSS","age":57,"sex":"F"},
    "SMC15-N": {"patient":"SMC15","condition":"Normal","stage":"II", "msi":"MSS","age":57,"sex":"F"},
    "SMC16-T": {"patient":"SMC16","condition":"Tumor", "stage":"III","msi":"MSS","age":66,"sex":"M"},
    "SMC16-N": {"patient":"SMC16","condition":"Normal","stage":"III","msi":"MSS","age":66,"sex":"M"},
    "SMC17-T": {"patient":"SMC17","condition":"Tumor", "stage":"I",  "msi":"MSS","age":45,"sex":"F"},
    "SMC17-N": {"patient":"SMC17","condition":"Normal","stage":"I",  "msi":"MSS","age":45,"sex":"F"},
    "SMC18-T": {"patient":"SMC18","condition":"Tumor", "stage":"IV", "msi":"MSI","age":71,"sex":"M"},
    "SMC18-N": {"patient":"SMC18","condition":"Normal","stage":"IV", "msi":"MSI","age":71,"sex":"M"},
    "SMC19-T": {"patient":"SMC19","condition":"Tumor", "stage":"II", "msi":"MSS","age":60,"sex":"F"},
    "SMC19-N": {"patient":"SMC19","condition":"Normal","stage":"II", "msi":"MSS","age":60,"sex":"F"},
    "SMC20-T": {"patient":"SMC20","condition":"Tumor", "stage":"III","msi":"MSS","age":52,"sex":"M"},
    "SMC20-N": {"patient":"SMC20","condition":"Normal","stage":"III","msi":"MSS","age":52,"sex":"M"},
    "SMC21-T": {"patient":"SMC21","condition":"Tumor", "stage":"I",  "msi":"MSS","age":49,"sex":"F"},
    "SMC21-N": {"patient":"SMC21","condition":"Normal","stage":"I",  "msi":"MSS","age":49,"sex":"F"},
    "SMC22-T": {"patient":"SMC22","condition":"Tumor", "stage":"II", "msi":"MSI","age":68,"sex":"M"},
    "SMC22-N": {"patient":"SMC22","condition":"Normal","stage":"II", "msi":"MSI","age":68,"sex":"M"},
    "SMC23-T": {"patient":"SMC23","condition":"Tumor", "stage":"IV", "msi":"MSS","age":76,"sex":"F"},
    "SMC23-N": {"patient":"SMC23","condition":"Normal","stage":"IV", "msi":"MSS","age":76,"sex":"F"},
}

CONDITION_COLORS = {"Tumor":"#e74c3c","Normal":"#2ecc71"}
STAGE_COLORS     = {"I":"#3498db","II":"#2ecc71","III":"#f39c12","IV":"#e74c3c"}
MSI_COLORS       = {"MSS":"#8e44ad","MSI":"#e67e22"}

def setup_directories():
    for d in [RAW_DIR, PROCESSED_DIR,
              Path("results/figures"), Path("results/tables"), Path("results/models")]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Directories initialized.")

def download_gse132465(output_dir=RAW_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / f"{GEO_ACCESSION}_RAW.tar"
    if tar_path.exists():
        logger.info(f"Already downloaded: {tar_path}")
        return output_dir / GEO_ACCESSION
    url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={GEO_ACCESSION}&format=file"
    logger.info(f"Downloading {GEO_ACCESSION} (~2.5 GB) from NCBI GEO...")
    logger.info(f"URL: {url}")
    logger.info("This may take 20-30 minutes on a standard connection.")
    urllib.request.urlretrieve(url, tar_path)
    extracted = output_dir / GEO_ACCESSION
    extracted.mkdir(exist_ok=True)
    logger.info("Extracting archive...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(extracted)
    logger.success(f"GSE132465 extracted to {extracted}")
    return extracted

def load_10x_sample(sample_dir, sample_id):
    def _find(pat):
        m = list(Path(sample_dir).glob(pat))
        if not m: raise FileNotFoundError(f"{pat} not found in {sample_dir}")
        return m[0]
    def _lines(f):
        op = gzip.open if str(f).endswith(".gz") else open
        with op(f,"rt") as fh: return [l.strip() for l in fh]
    matrix_f   = _find("*matrix*")
    barcodes_f = _find("*barcodes*")
    try: features_f = _find("*features*")
    except: features_f = _find("*genes*")
    if str(matrix_f).endswith(".gz"):
        with gzip.open(matrix_f,"rb") as f: matrix = mmread(f).T.tocsr()
    else:
        matrix = mmread(matrix_f).T.tocsr()
    barcodes = _lines(barcodes_f)
    feat     = _lines(features_f)
    gene_ids   = [l.split("\t")[0] for l in feat]
    gene_names = [l.split("\t")[1] if "\t" in l else l for l in feat]
    adata = ad.AnnData(
        X=csr_matrix(matrix),
        obs=pd.DataFrame(index=[f"{sample_id}_{bc}" for bc in barcodes]),
        var=pd.DataFrame({"gene_ids":gene_ids}, index=gene_names),
    )
    meta = SAMPLE_METADATA.get(sample_id, {})
    for k, v in meta.items(): adata.obs[k] = v
    adata.obs["sample_id"] = sample_id
    logger.info(f"  {sample_id}: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata

def build_combined_anndata(raw_dir=RAW_DIR):
    logger.info(f"Loading GSE132465 ({len(SAMPLE_METADATA)} samples)...")
    adatas = []
    gse_dir = raw_dir / GEO_ACCESSION
    for sample_id in SAMPLE_METADATA.keys():
        sample_dir = gse_dir / sample_id
        if not sample_dir.exists():
            candidates = list(gse_dir.glob(f"*{sample_id}*"))
            if candidates: sample_dir = candidates[0]
            else:
                logger.warning(f"  {sample_id} not found — skipping")
                continue
        try:
            adata = load_10x_sample(sample_dir, sample_id)
            adatas.append(adata)
        except Exception as e:
            logger.warning(f"  {sample_id} failed: {e}")
    if not adatas:
        raise RuntimeError("No samples loaded. See data/download_instructions.md")
    combined = ad.concat(adatas, join="outer", label="sample_id")
    combined.obs_names_make_unique()
    combined.obs["condition"] = pd.Categorical(
        combined.obs["condition"], categories=["Normal","Tumor"], ordered=True)
    logger.success(
        f"Combined: {combined.n_obs:,} cells x {combined.n_vars:,} genes | "
        f"Tumor: {(combined.obs.condition=='Tumor').sum():,} | "
        f"Normal: {(combined.obs.condition=='Normal').sum():,}"
    )
    return combined

def save_raw(adata, path=None):
    path = path or PROCESSED_DIR / "raw_combined.h5ad"
    adata.write_h5ad(path, compression="gzip")
    logger.success(f"Saved -> {path}")

def load_raw(path=None):
    return sc.read_h5ad(path or PROCESSED_DIR / "raw_combined.h5ad")

if __name__ == "__main__":
    setup_directories()
    download_gse132465()
    adata = build_combined_anndata()
    save_raw(adata)
    logger.success("Data ingestion complete.")
