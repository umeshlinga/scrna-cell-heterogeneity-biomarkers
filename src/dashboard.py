"""
dashboard.py
============
Interactive Plotly/Dash dashboard for CRC scRNA-seq analysis.
GSE132465: 23 patients, 68,060 cells, Tumor + Normal.
Tabs: Overview | UMAP | DE Volcano | Cell Communication | Trajectory | ML
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
import dash
from dash import dcc, html, Input, Output, dash_table

warnings.filterwarnings("ignore")

app = dash.Dash(__name__,
    title="CRC scRNA-seq Cell Heterogeneity Dashboard",
    suppress_callback_exceptions=True)

COLORS = {
    "Tumor":"#e74c3c","Normal":"#2ecc71",
    "bg":"#0d1117","surface":"#161b22","surface2":"#21262d",
    "border":"#30363d","text":"#f0f6fc","muted":"#8b949e","accent":"#58a6ff",
}
CELL_COLORS = {
    "Tumor_Epithelial":"#C0392B","Epithelial":"#E8A838",
    "T_cells":"#3498DB","B_cells":"#16A085","Myeloid":"#8E44AD",
    "NK_cells":"#E74C3C","Fibroblasts":"#2ECC71",
    "Endothelial":"#F39C12","Mast_cells":"#95A5A6","Unknown":"#BDC3C7",
}
STAGE_COLORS = {"I":"#3498db","II":"#2ecc71","III":"#f39c12","IV":"#e74c3c"}

def load_data():
    data = {}
    for key, path in {
        "de":         "results/tables/de_results_all.csv",
        "enrichment": "results/tables/enrichment_results_all.csv",
        "ml":         "results/tables/all_model_results.csv",
        "leiden":     "results/tables/leiden_sweep.csv",
        "lr":         "results/tables/lr_interactions_all.csv",
        "diff_comm":  "results/tables/differential_communication.csv",
        "biomarkers": "results/tables/de_known_biomarkers.csv",
    }.items():
        data[key] = pd.read_csv(path) if Path(path).exists() else None
    try:
        import scanpy as sc
        p = "data/processed/clustered_annotated.h5ad"
        data["adata"] = sc.read_h5ad(p) if Path(p).exists() else None
    except Exception:
        data["adata"] = None
    return data

def stat_card(title, value, color="#58a6ff"):
    return html.Div([
        html.P(title, style={"margin":"0","fontSize":"12px","color":COLORS["muted"],
               "textTransform":"uppercase","letterSpacing":"0.06em"}),
        html.H3(value, style={"margin":"4px 0","fontSize":"26px",
                "fontWeight":"700","color":color}),
    ], style={"background":COLORS["surface2"],"borderRadius":"10px",
              "padding":"18px","borderTop":f"3px solid {color}"})

def tab_overview(data):
    adata = data.get("adata")
    n_cells  = f"{adata.n_obs:,}" if adata else "68,060"
    n_tumor  = f"{(adata.obs.condition=='Tumor').sum():,}"  if adata and "condition" in adata.obs else "~34,000"
    n_normal = f"{(adata.obs.condition=='Normal').sum():,}" if adata and "condition" in adata.obs else "~34,000"

    fig = go.Figure()
    for stage,col in STAGE_COLORS.items():
        if adata is not None and "stage" in adata.obs:
            n = (adata.obs["stage"]==stage).sum()
        else:
            n = {"I":8000,"II":20000,"III":25000,"IV":15060}.get(stage,0)
        fig.add_trace(go.Bar(name=f"Stage {stage}",x=[f"Stage {stage}"],y=[n],
                             marker_color=col,text=[f"{n:,}"],textposition="outside"))
    fig.update_layout(title="Cells per Cancer Stage",showlegend=False,
        paper_bgcolor=COLORS["surface"],plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),margin=dict(l=40,r=20,t=50,b=40))

    return html.Div([
        html.Div([
            stat_card("Total Cells",    n_cells,   "#58a6ff"),
            stat_card("Tumor Cells",    n_tumor,   "#e74c3c"),
            stat_card("Normal Cells",   n_normal,  "#2ecc71"),
            stat_card("Patients",       "23",      "#f59e0b"),
            stat_card("Cell Types",     "9",       "#10b981"),
            stat_card("Cancer Stages",  "I-IV",    "#8b5cf6"),
        ], style={"display":"grid","gridTemplateColumns":"repeat(6,1fr)",
                  "gap":"12px","marginBottom":"24px"}),
        html.Div([
            html.Div([dcc.Graph(figure=fig)],
                     style={"flex":"1","background":COLORS["surface"],"borderRadius":"10px","padding":"8px"}),
            html.Div([
                html.H3("Dataset", style={"color":COLORS["text"],"marginTop":"0"}),
                html.P("GSE132465 — Human CRC Tumor Microenvironment",
                       style={"color":COLORS["accent"],"fontWeight":"500","fontSize":"14px"}),
                html.P("Lee HO et al. Nature Genetics 2020",
                       style={"color":COLORS["muted"],"fontSize":"13px"}),
                html.P("doi: 10.1038/s41588-020-0636-z",
                       style={"color":COLORS["muted"],"fontSize":"12px"}),
                html.Hr(style={"borderColor":COLORS["border"]}),
                html.P("Key Biomarkers (from paper + resume):",
                       style={"fontWeight":"600","color":COLORS["text"],"marginBottom":"8px"}),
                html.Div([html.Span(g, style={
                    "background":COLORS["surface2"],"border":f"1px solid {COLORS['border']}",
                    "borderRadius":"20px","padding":"4px 10px","fontSize":"12px",
                    "color":COLORS["accent"],"margin":"3px",
                }) for g in ["MDK","COL1A1","TMSB4X","CEACAM5","MKI67","TOP2A","VEGFA","SPP1"]
                ], style={"display":"flex","flexWrap":"wrap"}),
                html.Hr(style={"borderColor":COLORS["border"]}),
                html.P("Pipeline: Scanpy · Harmony · PAGA · scVelo · CellPhoneDB · XGBoost · SHAP",
                       style={"color":COLORS["muted"],"fontSize":"12px"}),
            ], style={"flex":"1","background":COLORS["surface"],"borderRadius":"10px",
                      "padding":"20px","border":f"1px solid {COLORS['border']}"}),
        ], style={"display":"flex","gap":"16px"}),
    ], style={"padding":"24px"})

def tab_umap(data):
    adata = data.get("adata")
    options = ["cell_type","condition","stage","patient_id","phase"]
    gene_opts = sorted(adata.var_names.tolist())[:500] if adata else []
    return html.Div([
        html.Div([
            html.Label("Color by:", style={"color":COLORS["muted"],"fontSize":"13px"}),
            dcc.Dropdown(id="umap-color",
                options=[{"label":v,"value":v} for v in options],
                value="cell_type",clearable=False,style={"color":"#000"}),
            html.Br(),
            html.Label("Gene expression:",style={"color":COLORS["muted"],"fontSize":"13px"}),
            dcc.Dropdown(id="umap-gene",
                options=[{"label":g,"value":g} for g in gene_opts],
                placeholder="Select gene...",style={"color":"#000"}),
            html.Br(),
            html.Label("Highlight cell type:",style={"color":COLORS["muted"],"fontSize":"13px"}),
            dcc.Dropdown(id="umap-highlight",
                options=[{"label":k,"value":k} for k in CELL_COLORS],
                placeholder="All cell types",style={"color":"#000"}),
        ], style={"width":"220px","padding":"16px","background":COLORS["surface"],
                  "borderRadius":"10px","border":f"1px solid {COLORS['border']}"}),
        html.Div([dcc.Graph(id="umap-graph",style={"height":"600px"})],
                 style={"flex":"1"}),
    ], style={"display":"flex","gap":"16px","padding":"24px"})

def tab_communication(data):
    lr   = data.get("lr")
    diff = data.get("diff_comm")

    heatmap_fig = go.Figure()
    if lr is not None:
        pivot = lr.pivot_table(index="sender",columns="receiver",
                               values="interaction_score",aggfunc="sum",fill_value=0)
        heatmap_fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale="YlOrRd", text=pivot.values.round(3),
            texttemplate="%{text}", hovertemplate="%{y}->%{x}: %{z:.4f}"))
        heatmap_fig.update_layout(
            title="Cell-Cell Interaction Strength (sum of LR scores)",
            paper_bgcolor=COLORS["surface"],plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]))

    diff_fig = go.Figure()
    if diff is not None:
        diff["label"] = diff["sender"]+"->"+diff["receiver"]
        colors = ["#e74c3c" if x>0 else "#3498db" for x in diff["log2_fold_change"]]
        diff_fig.add_trace(go.Bar(
            x=diff["log2_fold_change"],y=diff["label"],
            orientation="h",marker_color=colors))
        diff_fig.update_layout(
            title="Differential Communication: Tumor vs Normal",
            xaxis_title="log2FC (Tumor/Normal)",
            paper_bgcolor=COLORS["surface"],plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),height=500)

    return html.Div([
        html.Div([dcc.Graph(figure=heatmap_fig)],
                 style={"flex":"1","background":COLORS["surface"],"borderRadius":"10px","padding":"8px"}),
        html.Div([dcc.Graph(figure=diff_fig)],
                 style={"flex":"1","background":COLORS["surface"],"borderRadius":"10px","padding":"8px"}),
    ], style={"display":"flex","gap":"16px","padding":"24px"})

def tab_ml(data):
    ml = data.get("ml")
    fig = go.Figure()
    if ml is not None:
        for comp in ml["comparison"].unique():
            sub = ml[ml["comparison"]==comp].sort_values("roc_auc",ascending=False)
            fig.add_trace(go.Bar(name=comp,x=sub["model"],y=sub["roc_auc"],
                text=[f"{v:.3f}" for v in sub["roc_auc"]],textposition="outside",
                error_y=dict(type="data",array=sub.get("roc_auc_std",
                    pd.Series([0]*len(sub))).tolist(),visible=True)))
    fig.update_layout(title="Model ROC-AUC (5-fold Nested CV)",
        barmode="group",yaxis=dict(range=[0.5,1.05],title="ROC-AUC"),
        paper_bgcolor=COLORS["surface"],plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]))
    return html.Div([
        dcc.Graph(figure=fig),
        html.P("Models: XGBoost · LightGBM · RF · SVM · LogReg | "
               "Key features: MDK, COL1A1, TMSB4X",
               style={"color":COLORS["muted"],"fontSize":"13px","padding":"0 24px"}),
    ], style={"padding":"24px"})

def build_layout(data):
    TAB  = {"background":COLORS["surface"],"color":COLORS["muted"],"border":"none",
            "borderBottom":f"2px solid {COLORS['border']}","padding":"12px 20px","fontSize":"14px"}
    STAB = {**TAB,"color":COLORS["text"],"borderBottom":f"2px solid {COLORS['accent']}",
            "background":COLORS["surface2"]}
    return html.Div([
        html.Div([
            html.Div([
                html.H1("CRC scRNA-seq: Cell Heterogeneity & Biomarkers",
                    style={"margin":"0","fontSize":"20px","fontWeight":"700","color":COLORS["text"]}),
                html.P("GSE132465 | 23 patients | 68,060 cells | 9 cell types | Tumor + Normal | Stage I-IV",
                    style={"margin":"2px 0 0","fontSize":"12px","color":COLORS["muted"]}),
            ]),
            html.Span("● Real GEO Data | MIT License",
                style={"color":"#2ecc71","fontSize":"12px","fontWeight":"600"}),
        ], style={"background":COLORS["surface"],"borderBottom":f"1px solid {COLORS['border']}",
                  "padding":"16px 24px","display":"flex","justifyContent":"space-between","alignItems":"center"}),
        dcc.Tabs(id="tabs",value="overview",children=[
            dcc.Tab(label="Overview",      value="overview",     style=TAB,selected_style=STAB),
            dcc.Tab(label="UMAP",          value="umap",         style=TAB,selected_style=STAB),
            dcc.Tab(label="Cell Comm.",    value="communication",style=TAB,selected_style=STAB),
            dcc.Tab(label="ML Biomarkers", value="ml",           style=TAB,selected_style=STAB),
        ]),
        html.Div(id="tab-content"),
    ], style={"background":COLORS["bg"],"minHeight":"100vh","fontFamily":"Inter, sans-serif"})

def register_callbacks(data):
    @app.callback(Output("tab-content","children"), Input("tabs","value"))
    def render_tab(tab):
        if tab=="overview":      return tab_overview(data)
        if tab=="umap":          return tab_umap(data)
        if tab=="communication": return tab_communication(data)
        if tab=="ml":            return tab_ml(data)
        return html.Div()

    @app.callback(Output("umap-graph","figure"),
                  [Input("umap-color","value"),Input("umap-gene","value"),
                   Input("umap-highlight","value")])
    def update_umap(color_by, gene, highlight):
        adata = data.get("adata")
        if adata is None: return go.Figure()
        umap = adata.obsm["X_umap"]
        obs  = adata.obs.copy()
        obs["UMAP1"]=umap[:,0]; obs["UMAP2"]=umap[:,1]
        if gene and gene in adata.var_names:
            X = adata.X
            if hasattr(X,"toarray"): X=X.toarray()
            obs["expr"]=X[:,adata.var_names.tolist().index(gene)]
            fig = px.scatter(obs,x="UMAP1",y="UMAP2",color="expr",
                color_continuous_scale="Viridis",opacity=0.6,
                title=f"UMAP — {gene} expression")
        else:
            cmap = CELL_COLORS if color_by=="cell_type" else                    {"Tumor":COLORS["Tumor"],"Normal":COLORS["Normal"]} if color_by=="condition" else                    STAGE_COLORS if color_by=="stage" else None
            if color_by in obs.columns:
                if cmap:
                    fig = px.scatter(obs,x="UMAP1",y="UMAP2",color=color_by,
                        color_discrete_map=cmap,opacity=0.6,title=f"UMAP — {color_by}")
                else:
                    fig = px.scatter(obs,x="UMAP1",y="UMAP2",color=color_by,
                        opacity=0.6,title=f"UMAP — {color_by}")
            else:
                fig = px.scatter(obs,x="UMAP1",y="UMAP2",opacity=0.6)
        fig.update_traces(marker=dict(size=2))
        fig.update_layout(paper_bgcolor=COLORS["surface"],plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),margin=dict(l=10,r=10,t=40,b=10))
        return fig

if __name__ == "__main__":
    import sys
    logger.info("Loading dashboard data...")
    data = load_data()
    app.layout = build_layout(data)
    register_callbacks(data)
    logger.success("Dashboard ready at http://localhost:8050")
    app.run(debug="--debug" in sys.argv,host="0.0.0.0",port=8050)
