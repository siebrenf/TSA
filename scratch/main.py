import pandas as pd
import numpy as np
from tsa.preprocessing import get_sample_info, tpm_normalization
from tsa.gpr import gpr
from tsa.gene_selection import score_normalization, plot_scores, best_n_genes
from tsa.tsa import get_cost_matrix, best_alignment_graph, plot_alignment
from tsa.utils import list2floats, inference_timeseries
from tsa.plotting import plot_alignments


####################################################
# GPR
####################################################

# GPR input files
tpm_file = "data/GRCz11-TPM.tsv"
# template_samples_file = "data/white_stage_samples.tsv"
template_samples_file = "data/white_mpf_samples.tsv"

# variables
timepoints_per_sample = 10

# GPR output files
gpr_inference_file = "data/white_mpf_gpr.tsv"
gpr_scores_file = "data/white_mpf_score.tsv"

# gene selection
n_genes = 500
selected_genes_file = "data/white_stage_selected_genes.tsv"

####################################################

# preprocessing
template_samples = pd.read_csv(template_samples_file, sep="\t", index_col=0)
sample_order, time2samples = get_sample_info(template_samples)

tpms = pd.read_csv(tpm_file, sep="\t", index_col=0)
template_tpms = tpm_normalization(tpms, sample_order, minimum_value=5)

# filter genes BEFORE running GPR (optional)
selected_genes = list(pd.read_csv(selected_genes_file, sep="\t")["gene"])
template_tpms = template_tpms[template_tpms.index.isin(selected_genes)]

# GPR (slow)
extended_timepoints = list(np.round(np.linspace(min(time2samples), max(time2samples), 500), 2))
# extended_timepoints = inference_timeseries(list(time2samples), timepoints_per_sample)
template_tpms_inf, gpr_scores = gpr(time2samples, template_tpms, extended_timepoints, plot=False, verbose=False, run_n=None)
template_tpms_inf.to_csv(gpr_inference_file, sep="\t")
gpr_scores.to_csv(gpr_scores_file, sep="\t")

# # gene selection (optional)
# gpr_normscores = score_normalization(gpr_scores)
# plot_scores(gpr_normscores, highlight_top_n=n_genes)
# # save selected genes
# selected_genes = best_n_genes(gpr_normscores, n_genes=n_genes, to_file=selected_genes_file)


####################################################
# TSA
####################################################

# TSA input files
# tpm_file = "data/GRCz11-TPM.tsv"
# template_samples_file = "data/white_stage_samples.tsv"
# template_samples_file = "data/white_mpf_samples.tsv"
# gpr_inference_file = "data/white_stage_gpr.tsv"
# selected_genes_file = "data/white_stage_selected_genes.tsv"

# query_samples_file = "data/levin_stage_samples.tsv"
# query_samples_file = "data/white_stage_samples.tsv"
# query_samples_file = "data/marletaz_stage_samples.tsv"
# query_samples_file = "data/white_mpf_samples.tsv"
# query_samples_file = "data/levin_mpf_samples.tsv"
query_samples_file = "data/marletaz_mpf_samples.tsv"

# TSA output files
# alignment_file = "data/white_stage_levin_stage_mapping.tsv"
# alignment_file = "data/white_stage_white_stage_mapping.tsv"
# alignment_file = "data/white_stage_marletaz_stage_mapping.tsv"
# alignment_file = "data/white_mpf_white_mpf_mapping.tsv"
# alignment_file = "data/white_mpf_levin_mpf_mapping.tsv"
# alignment_file = "data/white_mpf_1000_white_mpf_mapping.tsv"
# alignment_file = "data/white_mpf_1000_levin_mpf_mapping.tsv"
alignment_file = "data/white_mpf_1000_marletaz_mpf_mapping.tsv"

####################################################

# preprocessing
query_samples = pd.read_csv(query_samples_file, sep="\t", index_col=0)
sample_order, time2samples = get_sample_info(query_samples)

tpms = pd.read_csv(tpm_file, sep="\t", index_col=0)
query_tpms = tpm_normalization(tpms, sample_order)

# cost matrix
# selected_genes = list(pd.read_csv(selected_genes_file, sep="\t")["gene"])
# template_tpms_inf = pd.read_csv(gpr_inference_file, sep="\t", index_col=0)
cost_matrix = get_cost_matrix(template_tpms_inf, query_tpms, selected_genes, time2samples)

# LTSA
best_path, best_score = best_alignment_graph(cost_matrix)
plot_alignment(cost_matrix, best_path)

# mapping
query_time = list2floats(query_samples.time.unique())
extended_template_time = list2floats(template_tpms_inf.columns)
mapped = pd.DataFrame(data={
    "original_time": query_time,
    "inferred_time": [extended_template_time[i] for i in best_path],
})
mapped.to_csv(alignment_file, sep="\t", index=False)  # noqa


####################################################
# Timeline
####################################################

# Timeline input files
# template_samples_file = "data/white_stage_samples.tsv"
# template_samples_file = "data/white_mpf_samples.tsv"
# gpr_inference_file = "data/white_mpf_gpr.tsv"

# alignment_files = {
#     "white": "data/white_stage_white_stage_mapping.tsv",
#     "levin": "data/white_stage_levin_stage_mapping.tsv",
#     "marletaz": "data/white_stage_marletaz_stage_mapping.tsv",
# }
# alignment_files = {
#     "white": "data/white_mpf_white_mpf_mapping.tsv",
#     "levin": "data/white_mpf_levin_mpf_mapping.tsv",
#     "marletaz": "data/white_mpf_marletaz_mpf_mapping.tsv",
# }
alignment_files = {
    "white": "data/white_mpf_1000_white_mpf_mapping.tsv",
    "levin": "data/white_mpf_1000_levin_mpf_mapping.tsv",
    "marletaz": "data/white_mpf_1000_marletaz_mpf_mapping.tsv",
}

####################################################

# x axis
# template_samples = pd.read_csv(template_samples_file, sep="\t", index_col=0)
# template_tpms_inf = pd.read_csv(gpr_inference_file, sep="\t", index_col=0)

template_time = list2floats(template_samples.time.unique())
extended_template_time = list2floats(template_tpms_inf.columns)
# extended_timepoints = inference_timeseries(template_time, timepoints_per_sample)
plot_alignments(template_time, extended_template_time, alignment_files)
