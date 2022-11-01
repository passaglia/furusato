import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from samplot.circusboy import CircusBoy
import samplot.colors as samcolors
import shutil
from matplotlib.ticker import FuncFormatter

import plotly.express as px
import plotly.graph_objects as go

from data import furusato_df, annual_df

PLOT_FOLDER = os.path.join(os.path.dirname(__file__), "ckzeffect/")

os.makedirs(PLOT_FOLDER, exist_ok=True)

oku = 10**8

cb = CircusBoy(
    baseFont="Helvetica",
    cjkFont="Hiragino Maru Gothic Pro",
    titleFont="Helvetica",
    textFont="Helvetica",
    fontsize=10,
    figsize=(5, 5 * 2 / 3),
)

################################
### For comparison to Glocal ###
################################


def compensationFinder(row):
    hoten = 0.75 * row.deductions
    # hoten = np.min([hoten, row['final-demand']- row['income']])
    print
    hoten = np.min([hoten, row["demand-pre-debt"] - row["income"]])
    hoten = np.max([hoten, 0])
    if (hoten != 0.75 * row.deductions) and hoten != 0:
        print("borderline city", row.city)
        print(row["demand-pre-debt"], row["income"])
        print(hoten)
        if row["prefecture"] == "東京都":
            hoten = 0
    return hoten


furusato_df["glocal-compensation"] = furusato_df.apply(compensationFinder, axis=1)
furusato_df["glocal-post-compensation"] = (
    furusato_df["netgainminusdeductions"] + furusato_df["glocal-compensation"]
)

#######################################################
### What fraction is from fukoufudantai? ##############
#######################################################
munidf = furusato_df.loc[(furusato_df["city"] != "prefecture")]
prefdf = furusato_df.loc[(furusato_df["city"] == "prefecture")]

fig, ax = cb.handlers()

ax.xaxis.set_major_formatter(FuncFormatter(r"{0:.0f}".format))
ax.yaxis.set_major_formatter(FuncFormatter(r"{0:.0f}\%".format))
cb.set_title(ax, r"ふるさと納税の寄付は交付団体からくる", r"ふるさと納税の控除額、交付団体対全自治体との割合", pad=18)

ax.plot(
    100
    * (
        munidf.loc[munidf["ckz"] > 0].groupby("year").sum()["deductions"]
        + prefdf.loc[prefdf["ckz"] > 0].groupby("year").sum()["deductions"]
    )
    / (
        munidf.groupby("year").sum()["deductions"]
        + prefdf.groupby("year").sum()["deductions"]
    )
)

for line in ax.lines:
    for x, y in zip(line.get_xdata(), line.get_ydata()):
        ax.annotate(
            r"{0:.0f}\%".format(y),
            xy=(x, y),
            xytext=(0, 6),
            color=line.get_color(),
            xycoords=(ax.get_xaxis_transform(), ax.get_yaxis_transform()),
            textcoords="offset points",
            size=cb.fontsize,
            ha="center",
            va="bottom",
        )

cb.set_byline(ax, "Sam Passaglia")
ax.set_xlim([2015.5, 2021.2])
ax.set_ylim([0, 102])
fig.savefig(PLOT_FOLDER + "fukoudantaifraction.pdf", bbox_inches="tight")

plt.close("all")

##################################################################################################
### Latex Table showing top FN losers and the compensation #####################################
##################################################################################################
df = furusato_df.loc[
    (furusato_df["city"] != "prefecture") & (furusato_df["year"] == 2021)
]
topN = 10

# Method 1: to latex
output_df = df.sort_values(by="netgainminusdeductions")[
    [
        "city",
        "netgainminusdeductions",
        "profit-from-ckz",
        "profit-from-special-debt",
        "profit-incl-ckz-incl-debt",
    ]
].head(topN)

## TODO: compare glocal-post-compensation to profit-incl-ckz-incl-debt

output_df = output_df.reset_index(drop=True)
output_df.index = output_df.index + 1


def formattermap(x):
    x = int(np.round(x / 10**8))
    if x < 0:
        return "\textcolor{red}{{%s}}" % x
    elif x >= 1:
        return "\textcolor{blue}{+{%s}}" % x
    else:
        return x


output_df[
    [
        "netgainminusdeductions",
        "profit-from-ckz",
        "profit-from-special-debt",
        "profit-incl-ckz-incl-debt",
    ]
] = output_df[
    [
        "netgainminusdeductions",
        "profit-from-ckz",
        "profit-from-special-debt",
        "profit-incl-ckz-incl-debt",
    ]
].applymap(
    formattermap
)

output_df.loc[
    output_df.index[0],
    [
        "netgainminusdeductions",
        "profit-from-ckz",
        "profit-from-special-debt",
        "profit-incl-ckz-incl-debt",
    ],
] = output_df.loc[
    output_df.index[0],
    [
        "netgainminusdeductions",
        "profit-from-ckz",
        "profit-from-special-debt",
        "profit-incl-ckz-incl-debt",
    ],
].str.replace(
    "}}", "億円}}"
)

colorstring = (
    r"\definecolor{blue}{rgb}{"
    + str(samcolors.nice_colors(3))[1:-1]
    + "}"
    + r"\definecolor{red}{rgb}{"
    + str(samcolors.nice_colors(0))[1:-1]
    + "}"
)

with open(PLOT_FOLDER + "table/table.tex", "w") as f:
    f.write(r"\documentclass{article}")
    f.write(r"\usepackage{booktabs}")
    f.write(r"\usepackage{makecell}")
    f.write(plt.rcParams["text.latex.preamble"])
    f.write(colorstring)
    f.write(
        r"""\begin{document}
    \hoffset=-1in
    \voffset=-1in
    \setbox0\hbox{"""
    )
    f.write(
        output_df.to_latex(
            column_format="c|l|r|r|r|r",
            header=[
                r"\textbf{自治体}",
                r"\makecell{\textbf{ふるさと納税赤字} \\\textbf{（都道府県分を除く）}}",
                r"\textbf{補塡寄付金}",
                r"\textbf{補塡地方債}",
                r"\textbf{赤字補塡後}",
            ],
            escape=False,
        )
    )
    f.write(
        r"""}
    \pdfpageheight=\dimexpr\ht0+\dp0\relax
    \pdfpagewidth=\wd0
    \shipout\box0\stop"""
    )


#################################################
### Where does the money come from #########################
#################################################

fig, ax = cb.handlers()

ax.plot(
    pd.to_datetime(annual_df.year, format="%Y"),
    100 * annual_df["profit-from-special-debt"] / (annual_df["donations"]),
    label="臨時債",
    color=samcolors.nice_colors(0),
)

ax.plot(
    pd.to_datetime(annual_df.year, format="%Y"),
    100
    * (
        annual_df["deductions"]
        - (annual_df["profit-from-special-debt"] + annual_df["profit-from-ckz"])
    )
    / (annual_df["donations"]),
    label=r"自治体",
    color=samcolors.nice_colors(3),
)

# TODO: split into uncompensated loss by fukoufudantai and others?

ax.plot(
    pd.to_datetime(annual_df.year, format="%Y"),
    100
    * (
        annual_df["donations"]
        - (annual_df["deductions"] - annual_df["profit-from-ckz"])
        - 2000 * annual_df["city-reported-people"]
    )
    / (annual_df["donations"]),
    label="国 (概算)",
    color=samcolors.nice_colors(2),
)

ax.plot(
    pd.to_datetime(annual_df.year, format="%Y"),
    100 * (2000 * annual_df["city-reported-people"]) / (annual_df["donations"]),
    label="寄付者",
    color=cb.grey,
)

for i, line in enumerate(ax.lines):
    # for x, y in zip(line.get_xdata(), line.get_ydata()):
    x, y = (line.get_xdata()[-1], line.get_ydata()[-1])
    s = line.get_label()
    x = matplotlib.dates.date2num(x)
    if i in [0]:
        xytext = (0, 4)
        ha = "center"
        va = "bottom"
    elif i in [1, 3]:
        xytext = (4, 1)
        ha = "left"
        va = "center"
    elif i in [2]:
        xytext = (0, -4)
        ha = "center"
        va = "top"
    ax.annotate(
        r"{0:.0f}\%".format(y) + s,
        xy=(x, y),
        xytext=xytext,
        color=line.get_color(),
        xycoords=(ax.get_xaxis_transform(), ax.get_yaxis_transform()),
        textcoords="offset points",
        size=cb.fontsize,
        ha=ha,
        va=va,
    )

ax.yaxis.set_major_formatter(FuncFormatter(r"{0:.0f}\%".format))
cb.set_xYear(ax)
cb.set_byline(ax, "サム・パッサリア")

colorstring = (
    r"\definecolor{blue}{rgb}{"
    + str(samcolors.nice_colors(3))[1:-1]
    + "}"
    + r"\definecolor{red}{rgb}{"
    + str(samcolors.nice_colors(0))[1:-1]
    + "}"
)
cb.titlesize = 16
cb.subtitlesize = 12
cb.set_title(
    ax,
    r"ふるさと納税、結局だれが払う？",
    # colorstring + r"\textcolor{blue}{ふるさと納税の寄付金}での減税を補填する\textcolor{red}{臨時財政債総額}",
    pad=0,
)

# ax.axvline(pd.to_datetime("2016-01-01"), color=cb.grey, zorder=-10, alpha=0.5,ls='--')
# ax.axvline(pd.to_datetime("2019-01-01"), color=cb.grey, zorder=-10, alpha=0.5,ls='--')

ax.set_ylim([0, 55])
ax.set_xlim([pd.to_datetime("2015-06-01"), pd.to_datetime("2021-12-01")])
fig.savefig(PLOT_FOLDER + "/total-debt.pdf")
plt.close("all")


#################################################
### Saitama ####################################
#################################################

# According to https://www.pref.saitama.lg.jp/a0103/saitamakensai/rinnjizaiseitaisakusai.html and comparing to the amount Saitama were allowed to issue in R2, they issued essentially the maximum amount they were allowed. But is there any way directly from my data to see that?
# https://www.pref.saitama.lg.jp/documents/209380/suii.pdf
fig, ax = cb.handlers()
prefectures = ["埼玉県"]
for prefecture in prefectures:
    prefdf = df.loc[df["prefecture"] == prefecture].sort_values("year")

    ax.plot(
        prefdf.year,
        prefdf["special-debt-payback"].values / 10**8,
        label="special-debt-payback",
        alpha=0.5,
    )
    ax.plot(
        prefdf.year,
        prefdf["special-debt"].values / 10**8,
        label="special-debt",
        ls="--",
        alpha=0.5,
    )
    ax.plot(
        prefdf.year,
        prefdf["special-debt"].values / 10**8 / 25,
        label="special-debt",
        ls="-.",
        alpha=0.5,
    )  ## Special-debt-payback is probably better called special-debt-servicing

ax.legend()
fig.savefig(PLOT_FOLDER + "/special-debt-payback-saitama.pdf")
plt.close("all")

# Others
# #https://www.pref.saitama.lg.jp/a0103/saitamakensai/rinnjizaiseitaisakusai.html
# fig, ax = cb.handlers()
# prefectures = np.random.choice(df['prefecture'].unique(), 10, replace=False)
# for prefecture in prefectures:
#     prefdf = df.loc[df['prefecture'] == prefecture].sort_values('year')

#     ax.plot(prefdf.year, prefdf['special-debt-payback'].values, label='special-debt-payback', alpha=.5)
#     ax.plot(prefdf.year, prefdf['special-debt'].values, label='special-debt', ls='--', alpha=.5)

# ax.legend()
# fig.savefig(PLOT_FOLDER+'/special-debt-payback.pdf')
# plt.close('all')

#######################################################
###### Total compensation (including seppan) ##########
#######################################################
munidf = (
    furusato_df.loc[(furusato_df["city"] != "prefecture")]
    .groupby("year")
    .sum()
    .reset_index()
)

prefdf = (
    furusato_df.loc[(furusato_df["city"] == "prefecture")]
    .groupby("year")
    .sum()
    .reset_index()
)

fig, ax = cb.handlers()
colorstring = (
    r"\definecolor{blue}{rgb}{"
    + str(samcolors.nice_colors(3))[1:-1]
    + "}"
    + r"\definecolor{red}{rgb}{"
    + str(samcolors.nice_colors(0))[1:-1]
    + "}"
)
cb.titlesize = 16
cb.subtitlesize = 12
cb.set_title(
    ax,
    r"ふるさと納税は地方交付税にどの影響？",
    "ふるさと納税で各自治体の減税に応じて、地方交付税での補填格。\n"
    + colorstring
    + r"\textcolor{red}{日経グローカルの概算}"
    + colorstring
    + r"と\textcolor{blue}{当分析}とは不一致。",
    pad=4,
)

ax.plot(
    munidf.year,
    (munidf["special-debt"] - munidf["special-debt-noFN"]) / 10**8,
    label="special-debt",
    ls="--",
    alpha=0.5,
)
ax.plot(
    munidf.year,
    (munidf["ckz"] - munidf["ckz-noFN"]) / 10**8,
    label="extra-ckz",
    ls="--",
    alpha=0.5,
)
ax.plot(
    munidf.year,
    (
        (munidf["special-debt"] - munidf["special-debt-noFN"])
        + (munidf["ckz"] - munidf["ckz-noFN"])
    )
    / 10**8,
    label="total",
    ls="--",
    alpha=0.5,
)
ax.plot(
    munidf.year,
    munidf["glocal-compensation"] / 10**8,
    label="glocal",
    ls="-",
    alpha=0.5,
)
fig.savefig(PLOT_FOLDER + "/compensation-muni.pdf")
plt.close("all")

fig, ax = cb.handlers()
ax.plot(
    prefdf.year,
    (prefdf["special-debt"] - prefdf["special-debt-noFN"]) / 10**8,
    label="special-debt",
    ls="--",
    alpha=0.5,
)
ax.plot(
    prefdf.year,
    (prefdf["ckz"] - prefdf["ckz-noFN"]) / 10**8,
    label="extra-ckz",
    ls="--",
    alpha=0.5,
)
ax.plot(
    prefdf.year,
    (
        (prefdf["special-debt"] - prefdf["special-debt-noFN"])
        + (prefdf["ckz"] - prefdf["ckz-noFN"])
    )
    / 10**8,
    label="total",
    ls="--",
    alpha=0.5,
)
ax.plot(
    prefdf.year,
    prefdf["glocal-compensation"] / 10**8,
    label="glocal",
    ls="-",
    alpha=0.5,
)
fig.savefig(PLOT_FOLDER + "/compensation-pref.pdf")
plt.close("all")

fig, ax = cb.handlers()
ax.plot(
    prefdf.year,
    (
        prefdf["special-debt"]
        - prefdf["special-debt-noFN"]
        + munidf["special-debt"]
        - munidf["special-debt-noFN"]
    )
    / 10**8,
    label="special-debt",
    ls="--",
    alpha=0.5,
)
ax.plot(
    prefdf.year,
    (prefdf["ckz"] - prefdf["ckz-noFN"] + munidf["ckz"] - munidf["ckz-noFN"]) / 10**8,
    label="extra-ckz",
    ls="--",
    alpha=0.5,
)
ax.plot(
    prefdf.year,
    (
        (prefdf["special-debt"] - prefdf["special-debt-noFN"])
        + (prefdf["ckz"] - prefdf["ckz-noFN"])
        + (munidf["special-debt"] - munidf["special-debt-noFN"])
        + (munidf["ckz"] - munidf["ckz-noFN"])
    )
    / 10**8,
    label="total",
    ls="--",
    alpha=0.5,
)
ax.plot(
    prefdf.year,
    (prefdf["glocal-compensation"] + munidf["glocal-compensation"]) / 10**8,
    label="glocal",
    ls="-",
    alpha=0.5,
)
fig.savefig(PLOT_FOLDER + "/compensation-total.pdf")
plt.close("all")

## Compare to nissei:
## Is she calculating only municipal or both sets of deductions when she says chihoukoufuzei compensates 2991oku  in 2022? I find glocal-style muni+pref computation leads to -> 3009oku


#################################################
### Debt scaling factors are NL in economic strength #####
#################################################


### Municipal ###
df = furusato_df.loc[
    (furusato_df["city"] != "prefecture") & (furusato_df["prefecture"] != "東京都")
]
year_df = df.loc[df["year"].isin([2021])]
fig = px.line(
    year_df,
    x="economic-strength-index-prev3yearavg",
    y="debt-scaling-factor",
    color="city",
    hover_data=["prefecture", "year", "economic-strength-index"],
    template="simple_white",
    markers=True,
)
fig.write_html(PLOT_FOLDER + "scaling-factors-muni.html")
plt.close("all")

### Prefectural ###
df = furusato_df.loc[
    (furusato_df["city"] == "prefecture") & (furusato_df["prefecture"] != "東京都")
]
year_df = df.loc[df["year"].isin([2021])]
fig = px.line(
    year_df,
    x="economic-strength-index-prev3yearavg",
    y="debt-scaling-factor",
    color="prefecture",
    hover_data=["prefecture", "year", "economic-strength-index"],
    template="simple_white",
    markers=True,
)
fig.write_html(PLOT_FOLDER + "scaling-factors-muni.html")
plt.close("all")


##################################################
### Debt scaling factors control ckz effect ######
##################################################

### Municipal
df = furusato_df.loc[
    (furusato_df["city"] != "prefecture") & (furusato_df["prefecture"] != "東京都")
]
df["dummy"] = (df["profit-from-ckz"] - df["deductions"]) / (
    df["demand-pre-debt"] - df["income"]
)
year_df = df.loc[df["year"].isin([2020, 2021])]
fig = px.line(
    year_df,
    x="debt-scaling-factor",
    y="dummy",
    color="city",
    hover_data=["prefecture", "year", "economic-strength-index"],
    template="simple_white",
    markers=True,
)
fig.write_html(PLOT_FOLDER + "debt-scaling-ckz-muni.html")
plt.close("all")

### Prefectural
df = furusato_df.loc[
    (furusato_df["city"] == "prefecture") & (furusato_df["prefecture"] != "東京都")
]
df["dummy"] = (df["profit-from-ckz"] - df["deductions"]) / (
    df["demand-pre-debt"] - df["income"]
)
year_df = df.loc[df["year"].isin([2020, 2021])]
fig = px.line(
    year_df,
    x="debt-scaling-factor",
    y="dummy",
    color="prefecture",
    hover_data=["prefecture", "year", "economic-strength-index"],
    template="simple_white",
    markers=True,
)
fig.write_html(PLOT_FOLDER + "debt-scaling-ckz.html")
plt.close("all")


##################################################
### Deductions vs ckz-compensation ###############
##################################################

### Municipal
df = furusato_df.loc[(furusato_df["city"] != "prefecture")]
df["dummy"] = (df["profit-from-ckz"] - df["deductions"]) / (
    df["demand-pre-debt"] - df["income"]
)
year_df = df.loc[df["year"].isin([2020])]

# plotly
# fig = px.line(year_df, x='deductions',y='profit-from-ckz', color='city',hover_data=['prefecture', 'year','economic-strength-index'], template='simple_white',markers=True)
# fig.write_html(PLOT_FOLDER+'ckz-profit-vs-deductions-muni.html')
# plt.close('all')

# mpl
fig, ax = cb.handlers()
year_df = year_df.loc[year_df["ckz"] > 0]
ax.scatter(
    year_df["deductions"] / 10**8,
    year_df["glocal-compensation"] / 10**8,
    color=samcolors.nice_colors(0),
    alpha=0.5,
)
ax.scatter(
    year_df["deductions"] / 10**8,
    year_df["profit-from-ckz"] / 10**8,
    color=samcolors.nice_colors(3),
    alpha=0.5,
)
cb.set_yunit(ax, "億円が補填されている")
cb.set_xunit(ax, "億円")
cb.set_byline(ax, "サム・パッサリア")
colorstring = (
    r"\definecolor{blue}{rgb}{"
    + str(samcolors.nice_colors(3))[1:-1]
    + "}"
    + r"\definecolor{red}{rgb}{"
    + str(samcolors.nice_colors(0))[1:-1]
    + "}"
)
cb.titlesize = 16
cb.subtitlesize = 12
cb.set_title(
    ax,
    r"ふるさと納税は地方交付税にどの影響？",
    "ふるさと納税で各自治体の減税に応じて、地方交付税での補填格。\n"
    + colorstring
    + r"\textcolor{red}{日経グローカルの概算}"
    + colorstring
    + r"と\textcolor{blue}{当分析}とは不一致。",
    pad=4,
)
ax.set_xlabel("ふるさと納税での控除格")

x = 91.76
y = x * 0.75
ax.annotate(
    "日経グローカルの概算では\n控除の３／４が補填にされる",
    xy=(x, y),
    xytext=(5, 67),
    arrowprops=dict(
        arrowstyle="->",
        shrinkA=0,
        shrinkB=5,
        color=samcolors.nice_colors(0),
        connectionstyle="angle3,angleA=25,angleB=-25",
    ),
    fontsize=10,
    color=samcolors.nice_colors(0),
    bbox=dict(boxstyle="round", fc="w", ec="None"),
)
x = 55
y = -14
ax.annotate(
    "しかし、交付税の総額が固定で補填がゼロサム。\n 当分析は地方交付税の総合的なシミュレーション\nを実行する。",
    xy=(x, y),
    xytext=(65, -15),
    arrowprops=dict(
        arrowstyle="->",
        shrinkA=0,
        shrinkB=0,
        color=samcolors.nice_colors(3),
        connectionstyle="angle3,angleA=-15,angleB=+15",
    ),
    fontsize=10,
    color=samcolors.nice_colors(3),
    bbox=dict(boxstyle="round", fc="w", ec="None"),
)

fig.savefig(PLOT_FOLDER + "ckz-profit-vs-deductions-muni.pdf")
plt.close("all")

### Prefectural
df = furusato_df.loc[
    (furusato_df["city"] == "prefecture") & (furusato_df["prefecture"] != "東京都")
]
# df = furusato_df.loc[(furusato_df['city']=='prefecture')]
df["dummy"] = (df["profit-from-ckz"] - df["deductions"]) / (
    df["demand-pre-debt"] - df["income"]
)
year_df = df.loc[df["year"].isin([2021])]

# plotly
fig = px.line(
    year_df,
    x="deductions",
    y="profit-from-ckz",
    color="prefecture",
    hover_data=["prefecture", "year", "economic-strength-index"],
    template="simple_white",
    markers=True,
)
fig.write_html(PLOT_FOLDER + "ckz-profit-vs-deductions.html")
plt.close("all")

# mpl
fig, ax = cb.handlers()
ax.scatter(
    year_df["deductions"],
    np.where(
        year_df["glocal-compensation"].values > 0,
        0.75 * year_df["deductions"].values,
        0,
    ),
)
ax.scatter(year_df["deductions"], year_df["profit-from-ckz"])
fig.savefig(PLOT_FOLDER + "ckz-profit-vs-deductions.pdf")
plt.close("all")
