import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# universal time axis
all_mpf = [
    0,
    45,
    60,
    75,
    90,
    105,
    120,
    135,
    150,
    165,
    180,
    200,
    220,
    240,
    260,
    280,
    315,
    340,
    360,
    480,
    540,
    600,
    620,
    700,
    840,
    960,
    1140,
    1320,
    1440,
    1800,
    2160,
    2520,
    2880,
    3600,
    4320,
    5760,
    7200,
    8640,
    10080,
    20160,
    30240,
    43200,
    64800,
][:-6]
all_stages = [
    "1-cell",
    "2-cell",
    "4-cell",
    "8-cell",
    "16-cell",
    "32-cell",
    "64-cell",
    "128-cell",
    "256-cell",
    "512-cell",
    "1k-cell",
    "High",
    "Oblong",
    "Sphere",
    "Dome",
    "30%-epiboly",
    "50%-epiboly",
    "Germ-ring",
    "Shield",
    "75%-epiboly",
    "90%-epiboly",
    "Bud",
    "1-4 somites",
    "5-9 somites",
    "10-13 somites",
    "14-19 somites",
    "20-25 somites",
    "26+ somites",
    "Prim-5",
    "Prim-15",
    "Prim-25",
    "High-pec",
    "Long-pec",
    "Pec-fin",
    "Protruding-mouth",
    "Day 4",
    "Day 5",
    "Day 6",
    "Days 7-13",
    "Days 14-20",
    "Days 21-29",
    "Days 30-44",
    "Days 45-89",
][:-6]

# query alignments
# alignments = {
#     "white": "data/white_mpf_white_mpf_mapping.tsv",
#     "levin": "data/white_mpf_levin_mpf_mapping.tsv",
#     "marletaz": "data/white_mpf_marletaz_mpf_mapping.tsv",
# }
alignments = {
    "white": "data/white_mpf_1000_white_mpf_mapping.tsv",
    "levin": "data/white_mpf_1000_levin_mpf_mapping.tsv",
    "marletaz": "data/white_mpf_1000_marletaz_mpf_mapping.tsv",
}

plt.rcParams['figure.figsize'] = [24, 8]
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

n = 0
ylabels = []
for series in alignments:
    df = pd.read_csv(alignments[series], sep="\t")
    x1 = df["original_time"]
    x2 = df["inferred_time"]
    plt.scatter(x=x1, y=np.zeros_like(x1) + n)
    n += 1
    plt.scatter(x=x2, y=np.zeros_like(x2) + n)
    n += 1
    ylabels.extend([f"{series} annotated", f"{series} inferred"])

# plot shape
x_axis = all_mpf
x_range = max(x_axis) - min(x_axis)
plt.yticks(list(range(n)), ylabels)
plt.ylim(-0.5, n - 0.5)
plt.ylabel("time series")

total_time = max(x_axis) - min(x_axis)
plt.xlim(min(x_axis) - x_range * 0.03, max(x_axis) + x_range * 0.03)

ax1.set_xticks(x_axis)
ax1.set_xticklabels(all_stages, rotation=45, ha="right")
ax1.set_xlabel("developmental stage")

ax2.set_xticks(x_axis)
ax2.set_xticklabels(all_mpf, rotation=45, ha="left")
ax2.set_xlabel("minutes post fertilization")

plt.show()



#########################################



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tsa.utils import inference_timeseries
#
#
# # universal time axis
# all_mpf = [
#     0,
#     45,
#     60,
#     75,
#     90,
#     105,
#     120,
#     135,
#     150,
#     165,
#     180,
#     200,
#     220,
#     240,
#     260,
#     280,
#     315,
#     340,
#     360,
#     480,
#     540,
#     600,
#     620,
#     700,
#     840,
#     960,
#     1140,
#     1320,
#     1440,
#     1800,
#     2160,
#     2520,
#     2880,
#     3600,
#     4320,
#     5760,
#     7200,
#     8640,
#     10080,
#     20160,
#     30240,
#     43200,
#     64800,
# ][:-6]
# all_stages = [
#     "1-cell",
#     "2-cell",
#     "4-cell",
#     "8-cell",
#     "16-cell",
#     "32-cell",
#     "64-cell",
#     "128-cell",
#     "256-cell",
#     "512-cell",
#     "1k-cell",
#     "High",
#     "Oblong",
#     "Sphere",
#     "Dome",
#     "30pc-epiboly",
#     "50pc-epiboly",
#     "Germ-ring",
#     "Shield",
#     "75pc-epiboly",
#     "90pc-epiboly",
#     "Bud",
#     "1-4-somites",
#     "5-9-somites",
#     "10-13-somites",
#     "14-19-somites",
#     "20-25-somites",
#     "26+-somites",
#     "Prim-5",
#     "Prim-15",
#     "Prim-25",
#     "High-pec",
#     "Long-pec",
#     "Pec-fin",
#     "Protruding-mouth",
#     "Day 4",
#     "Day 5",
#     "Day 6",
#     "Days 7-13",
#     "Days 14-20",
#     "Days 21-29",
#     "Days 30-44",
#     "Days 45-89",
# ][:-6]
#
# # template time (in case time is not numeric)
# # template_samples_file = "data/white_mpf_samples.tsv"
# template_samples_file = "data/white_stage_samples.tsv"
# timepoints_per_sample = 10
# template_samples = pd.read_csv(template_samples_file, sep="\t", index_col=0)
# template_time = list(template_samples.time.unique())
# extended_template_time = inference_timeseries(template_time, timepoints_per_sample)
#
# ext_time_axis = pd.DataFrame({
#     "ext_all_mpf": inference_timeseries(all_mpf, timepoints_per_sample),
#     "ext_all_stages": inference_timeseries(all_stages, timepoints_per_sample),
# })
#
# # query alignments
# alignments = {
#     "white": "data/white_stage_white_stage_mapping.tsv",
#     "levin": "data/white_stage_levin_stage_mapping.tsv",
#     "marletaz": "data/white_stage_marletaz_stage_mapping.tsv",
# }
#
# plt.rcParams['figure.figsize'] = [24, 8]
# fig = plt.figure(1)
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twiny()
#
# n = 0
# ylabels = []
# for series in alignments:
#     df = pd.read_csv(alignments[series], sep="\t")
#
#     x1 = df["original_time"]
#     if df["original_time"].dtype == "O":
#         cv = ext_time_axis.merge(df, left_on="ext_all_stages", right_on="original_time", how="right")
#         x1 = cv["ext_all_mpf"].to_list()
#     plt.scatter(x=x1, y=np.zeros_like(x1) + n)
#     n += 1
#
#     x2 = df["inferred_time"]
#     if df["inferred_time"].dtype == "O":
#         cv = ext_time_axis.merge(df, left_on="ext_all_stages", right_on="inferred_time", how="right")
#         x2 = cv["ext_all_mpf"].to_list()
#     plt.scatter(x=x2, y=np.zeros_like(x2) + n)
#     n += 1
#     ylabels.extend([f"{series} annotated", f"{series} inferred"])
#
# # plot shape
# x_axis = all_mpf
# x_range = max(x_axis) - min(x_axis)
# plt.yticks(list(range(n)), ylabels)
# plt.ylim(-0.5, n - 0.5)
# plt.ylabel("time series")
#
# # total_time = max(x_axis) - min(x_axis)
# # plt.xlim(min(x_axis), max(x_axis))
# plt.xlim(min(x_axis) - x_range * 0.03, max(x_axis) + x_range * 0.03)
# # plt.ylim(-0.5, 2.5)
#
# ax1.set_xticks(x_axis)
# ax1.set_xticklabels(all_stages, rotation=45, ha="right")
# ax1.set_xlabel("developmental stage")
#
# ax2.set_xticks(x_axis)
# ax2.set_xticklabels(all_mpf, rotation=45, ha="left")
# ax2.set_xlabel("minutes post fertilization")
#
# plt.show()
