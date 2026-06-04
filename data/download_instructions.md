# Dataset Download Instructions

## GSE132465 — Human Colorectal Cancer scRNA-seq

### Paper
Lee HO et al. Lineage-dependent gene expression programs influence
the init response to tumor invasion. **Nature Genetics** 2020.
doi: 10.1038/s41588-020-0636-z

### Dataset Details
- Cells: 68,060 single cells
- Patients: 23 CRC patients
- Conditions: Tumor + Adjacent Normal
- Cancer stages: I, II, III, IV
- Platform: 10x Genomics Chromium
- Organism: Homo sapiens

### Download Steps
1. Visit: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132465
2. Scroll to Supplementary files
3. Download: GSE132465_RAW.tar.gz
4. Extract to: data/raw/GSE132465/

### Expected Structure
data/raw/GSE132465/
  SMC01-T/  (Patient 1 Tumor)
  SMC01-N/  (Patient 1 Normal)
  SMC02-T/
  ...
  SMC23-N/

### Cell Types (from original paper)
- Epithelial cells (tumor + normal)
- T cells (CD8+, CD4+, Treg)
- B cells
- Myeloid cells (macrophages, DCs, monocytes)
- NK cells
- Fibroblasts (CAFs, normal)
- Endothelial cells
- Mast cells
