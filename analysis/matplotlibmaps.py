import os
import pandas as pd
import numpy as np
import shutil
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from samplot.utils import init_plotting
from samplot.baseplot import BasePlot
from samplot.circusboy import CircusBoy

import samplot.colors as samcolors

from data import local_df, pref_df
from japandata.maps.data import add_df_to_map

from config import productionPlotFolder

PLOT_FOLDER = os.path.join(os.getcwd(), "matplotlibmaps/")

output_filetypes = ["pdf"]

os.makedirs(PLOT_FOLDER, exist_ok=True)

hyakuman = 10**6

year = 2021

pref_df = pref_df.loc[pref_df["year"] == year]
local_df = local_df.loc[local_df["year"] == year]

################################
#### Loading map ###
#################################
pref_map_df = add_df_to_map(pref_df, date=year, level="prefecture", quality="stylized")

##################################
### Matplotlib version -- pref ###
##################################

df = pref_map_df.copy()

datacolumn = "profit-per-person"
# datacolumn= "profit-per-person-incl-ckz"

scalingfactor = 1
roundinglevel = 1000
df["_dummy"] = (df[datacolumn] / scalingfactor / roundinglevel).apply(
    np.round
) * roundinglevel

import shapely

print("moving okinawa")
df.loc[df["prefecture"] == "沖縄県", "geometry"] = df.loc[
    df["prefecture"] == "沖縄県", "geometry"
].affine_transform([1, 0, 0, 1, 6.5, 13])

print("rotating")
rotation_angle = -17
rotation_origin = df[df.is_valid].unary_union.centroid
df["geometry"] = df["geometry"].rotate(rotation_angle, origin=rotation_origin)


titles = {
    "en": r"\textbf{Winning and losing prefectures}"
    + "\n"
    + r"\textbf{"
    + str(year)
    + "}",
    "jp": r"\textbf{儲かっている、損している都道府県}" + "\n" + r"\textbf{" + str(year) + "}",
}  # 都道府県での勝ち組と負け組
labels = {"en": r"Profit per capita (\textyen/person)", "jp": r"一人当たりのふるさと納税純利益（円／人）"}
df = df.to_crs("EPSG:30166")

for lang in titles.keys():
    bp = BasePlot(figsize=(6, 6), fontsize=12, textFont="Helvetica")
    fig, ax = bp.handlers()
    plt.axis("off")

    # from matplotlib.colors import TwoSlopeNorm

    bins = [-10001, -6000, -1000, 0, 1000, 6000, 10001, 21001]
    bluecolors = matplotlib.cm.get_cmap("coolwarm")(np.linspace(0, 0.45, 3))
    redcolors = matplotlib.cm.get_cmap("coolwarm")(np.linspace(0.55, 1, 4))
    rgbacolors = np.concatenate([bluecolors, redcolors])
    from matplotlib.colors import ListedColormap

    cmap, norm = matplotlib.colors.from_levels_and_colors(
        bins, rgbacolors, extend="neither"
    )

    ax = df.plot(
        column="_dummy",
        ax=ax,
        cmap=cmap,
        norm=norm,
        legend=False,
        lw=0.05,
        edgecolor="#4341417c",
    )
    # national_map_df.plot(ax=ax,edgecolor=(0,0,0),facecolor="None",lw=.3)
    # df.loc[df['prefecture'] == "沖縄県"].plot(ax=ax,edgecolor=(0,0,0),facecolor="None",lw=.3)#scheme='userdefined', edgecolor=(0,0,0),lw=.1, classification_kwds={'bins':bins}
    ax.set_xlim([-0.75 * 10**6, 0.98 * 10**6])
    ax.set_ylim([-0.28 * 10**6, 0.95 * 10**6])

    fig.suptitle(
        titles[lang],
        x=0,
        y=1.0,
        fontsize=bp.titlesize,
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import FuncFormatter, StrMethodFormatter

    divider = make_axes_locatable(ax)

    cbwidth = ax.get_position().width * 0.75
    axwidth = ax.get_position().width
    xstart = ax.get_position().x0 + (axwidth - cbwidth) / 2
    cbheight = 0.015
    cax = fig.add_axes([xstart, ax.get_position().y0 - cbheight, cbwidth, cbheight])

    cb = matplotlib.colorbar.Colorbar(
        cax,
        cmap=cmap,
        norm=norm,
        boundaries=np.round(np.array(bins) / roundinglevel) * roundinglevel,
        extend="neither",
        ticks=np.round(np.array(bins) / roundinglevel) * roundinglevel,
        # spacing='proportional',
        spacing="uniform",
        orientation="horizontal",
    )

    cax.xaxis.set_major_formatter(FuncFormatter(r"{0:.0f}".format))
    cb.ax.tick_params(axis="both", colors="none")
    cb.outline.set_edgecolor("none")
    cb.set_label(labels[lang], color="black")

    # https://medium.datadriveninvestor.com/creating-a-discrete-colorbar-with-custom-bin-sizes-in-matplotlib-50b0daf8dd46
    # https://stackoverflow.com/questions/36008648/colorbar-on-geopandas

    pc = [-1 * 10**4, 4.2 * 10**5]
    L = 1.2 * 10**5
    phi = 35 * np.pi / 180  ## angle to vertical
    theta = 140 * np.pi / 180  ## opening angle
    pR = [L * np.sin(phi) + pc[0], L * np.cos(phi) + pc[1]]
    pL = [
        pc[0] - L * np.cos(np.pi / 2 - theta + phi),
        pc[1] + L * np.sin(np.pi / 2 - theta + phi),
    ]
    ax.plot([pL[0], pc[0], pR[0]], [pL[1], pc[1], pR[1]], color="black", lw=1)

    fig.savefig(
        PLOT_FOLDER + "prefecture-profit_" + lang + ".pdf",
        transparent=True,
        bbox_inches="tight",
    )
    shutil.copy(
        PLOT_FOLDER + "prefecture-profit_" + lang + ".pdf", productionPlotFolder
    )

    plt.close("all")

print(len(df.loc[df["profit-per-person"] < 0]))
print(len(df.loc[df["profit-per-person-incl-ckz"] < 0]))


#############################
#### For each prefecture ####
#############################
import contextily as cx
from xyzservices import TileProvider

from config import mapboxToken, tileCacheDirectory

cx.set_cache_dir(tileCacheDirectory)

oku = 10**8
hyakuman = 10**6


def make_pref(PREFECTURE):

    print(PREFECTURE)
    PLOT_FOLDER = os.path.join(os.getcwd(), "singlepref/" + PREFECTURE + "/")
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    #### Restrict to one prefecture

    pref_map_df = add_df_to_map(
        pref_df, date=year, level="prefecture", quality="medium"
    )

    local_map_df = add_df_to_map(
        local_df, date=year, level="local_dc", quality="medium"
    )

    local_map_df = local_map_df.loc[local_map_df["_merge"] == "both"]

    pref_map_df = pref_map_df.loc[(pref_map_df["prefecture"] == PREFECTURE)].copy()

    df = local_map_df.loc[(local_map_df["prefecture"] == PREFECTURE)].copy()

    tokyo23codes = [str(13101 + i) for i in range(23)]
    df = df.drop(df.loc[df["code"].isin(tokyo23codes)].index)

    df = df.reset_index(drop=True)

    ##################################
    ### Matplotlib map ###########
    ##################################
    datacolumn = "profit-incl-ckz"
    datacolumn = "netgainminusdeductions"
    # datacolumn = "profit-incl-ckz-incl-pref-tax-share"

    maxval = df[datacolumn].max() / oku
    if maxval <= 5:
        maxbin = np.ceil(maxval)
        bins = np.arange(0, maxbin + 1, 1)
    elif 5 <= maxval <= 10:
        bins = [0, 1, 5, 10]
    elif maxval < 60:
        maxbin = np.ceil(maxval / 10) * 10
        bins = [0, 1, 5] + list(np.arange(10, maxbin + 10, 10))
    else:
        maxbin = np.ceil(maxval / 10) * 10
        bins = np.ones(11)
        step = 10
        while len(bins) > 7:
            bins = [0, 1, 5] + list(np.arange(10, maxbin + 10, step))
            if bins[-1] < maxbin:
                bins[-1] = maxbin
            step += 10

    if maxval > 1:
        bins[-1] = np.ceil(maxval)
    else:
        bins[-1] = np.ceil(maxval * 100) / 100
    bins = np.array(bins) * oku

    largenegativenumber = -1000 * oku

    print("Total number of municipalities:", len(df[datacolumn]))
    hist, histbins = np.histogram(
        df[datacolumn], np.array([largenegativenumber] + list(bins))
    )
    assert hist.sum() == len(df[datacolumn])
    for i in range(len(histbins) - 1):
        print(
            str(hist[i])
            + " municipalities ( "
            + str(round(hist[i] / hist.sum() * 100))
            + " \% ) in range "
            + str(histbins[i] / oku)
            + "-"
            + str(histbins[i + 1] / oku),
            "oku yen",
        )
        print(
            df.loc[
                (histbins[i] < df[datacolumn]) & (df[datacolumn] < histbins[i + 1]),
                [
                    "city-reading",
                    "total-pop",
                    "netgainminusdeductions",
                    "donations",
                    # "donations-fraction",
                    "economic-strength-index",
                ],
            ].sort_values("netgainminusdeductions")
        )

    scalingfactors = {"en": hyakuman, "jp": oku}

    PREFECTURE_EN = (
        (pref_map_df["prefecture-reading"].iloc[0])
        .replace("ou", "o")
        .replace("oo", "o")
        .title()
    )

    langs = ["en", "jp"]
    title_strings = {
        "en": r"\textbf{Furusato n\={o}zei profit in "
        + PREFECTURE_EN
        + r"  in "
        + str(year)
        + "}",
        "jp": r"\textbf{" + PREFECTURE + "におけるふるさと納税純利益（" + str(year) + "年）}",
    }
    xlabels = {"en": r"Profit (\textyen mn)", "jp": r"純利益 (億円)"}
    losslabels = {"en": r"Loss", "jp": r"損失"}
    labelproviders = {
        "en": TileProvider(
            {
                "name": "Mapbox Labels Only En",
                "attribution": "",
                "url": "https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}{r}?access_token={accessToken}",
                "accessToken": mapboxToken,
                "id": "mithridatesvi/cl4xdjlvd000315myqsnkyhn7",
                "r": "@2x",
            }
        ),
        "jp": TileProvider(
            {
                "name": "Mapbox Labels Only Jp",
                "attribution": "",
                "url": "https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}{r}?access_token={accessToken}",
                "accessToken": mapboxToken,
                "id": "mithridatesvi/cl64uc0sx005114s475i57a01",
                "r": "@2x",
            }
        ),
    }

    for lang in langs:

        df["_dummy"] = df[datacolumn] / scalingfactors[lang]

        dpi = 400
        if PREFECTURE == "北海道":
            dpi = 300

        bp = BasePlot(figsize=(6, 6), fontsize=12, dpi=dpi)
        fig, ax = bp.handlers()
        plt.axis("off")
        fig.suptitle(
            title_strings[lang],
            x=0,
            y=1.01,
            fontsize=bp.titlesize,
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            wrap=True,
        )
        # t = fig.suptitle(title_strings[lang], x=0.05,y=.95, fontsize=16,ha='left',va='top', transform=ax.transAxes, wrap=True,bbox=dict(facecolor='white', alpha=1, edgecolor='white'))
        # , boxstyle='round,pad=.1'

        if PREFECTURE != "北海道":
            pref_map_df.plot(ax=ax, edgecolor=(0, 0, 0), facecolor="None", lw=0.3)

        from matplotlib.colors import ListedColormap, Normalize, from_levels_and_colors

        blue = samcolors.nice_colors(3)
        blue = np.array(list(blue) + [1])
        redcolors = matplotlib.cm.get_cmap("coolwarm")(
            np.linspace(0.5, 1, len(bins) - 1)
        )
        colors = [blue] + list(redcolors)
        alpha = 1
        for i in range(len(colors)):
            colors[i][-1] = alpha

        cmap, norm = from_levels_and_colors(
            np.array([largenegativenumber] + list(bins)) / scalingfactors[lang],
            colors,
            extend="neither",
        )
        df.to_crs(epsg=3857).plot(
            column="_dummy",
            ax=ax,
            cmap=cmap,
            norm=norm,
            legend=False,
            edgecolor=(0, 0, 0),
            lw=0.1,
        )

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="2%", pad=0.2)
        cb2 = matplotlib.colorbar.ColorbarBase(
            cax,
            cmap=cmap,
            norm=norm,
            boundaries=np.array([-oku] + list(bins)) / scalingfactors[lang],
            extend="neither",
            ticks=bins / scalingfactors[lang],
            # spacing='proportional',
            spacing="uniform",
            orientation="horizontal",
        )
        from matplotlib.ticker import FuncFormatter, StrMethodFormatter

        cax.xaxis.set_major_formatter(FuncFormatter(r"{0:.0f}".format))
        cax.text(
            largenegativenumber / scalingfactors[lang] / 2,
            1.5,
            losslabels[lang],
            ha="center",
        )
        cb2.ax.tick_params(axis="both", colors="none")
        cb2.outline.set_edgecolor("none")
        cb2.set_label(xlabels[lang])

        mainprovider = TileProvider(
            {
                "name": "Mapbox Background Only",
                "attribution": "",
                "url": "https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}{r}?access_token={accessToken}",
                "accessToken": mapboxToken,
                "id": "mithridatesvi/cl4gf57r8000614mc0xrwsvt7",
                "r": "@2x",
            }
        )

        labelprovider = labelproviders[lang]

        xBuffer = 0.5
        yBuffer = 0.5
        boundsarr = df[df.is_valid].unary_union.bounds
        center = df[df.is_valid].unary_union.centroid.coords.xy
        yL = boundsarr[3] - boundsarr[1]

        reaspect = True
        desired_aspect_ratio = 4 / 3
        if reaspect:
            boundsarr = [
                center[0][0] - yL / 2 * desired_aspect_ratio,
                boundsarr[1],
                center[0][0] + yL / 2 * desired_aspect_ratio,
                boundsarr[3],
            ]

        xL = boundsarr[2] - boundsarr[0]

        suggested_zoom = cx.tile._calculate_zoom(
            boundsarr[0] - xBuffer * xL,
            boundsarr[1] - yBuffer * yL,
            boundsarr[2] + xBuffer * xL,
            boundsarr[3] + yBuffer * yL,
        )

        labels_img, labels_ext = cx.bounds2img(
            *df[df.is_valid].unary_union.bounds,
            ll=True,
            zoom=suggested_zoom - 1,
            source=labelprovider
        )

        ax.imshow(labels_img, extent=labels_ext, zorder=+10, interpolation="sinc")

        img, ext = cx.bounds2img(
            boundsarr[0] - xBuffer * xL,
            boundsarr[1] - yBuffer * yL,
            boundsarr[2] + xBuffer * xL,
            boundsarr[3] + yBuffer * yL,
            ll=True,
            source=mainprovider,
            zoom=suggested_zoom,
        )
        ax.imshow(img, extent=ext)

        xBuffer = 0.02
        yBuffer = 0.01

        boundsarr = pref_map_df.to_crs(epsg=3857).bounds.values[0]
        center = pref_map_df.to_crs(epsg=3857).centroid.iloc[0].coords.xy

        if PREFECTURE in ["東京都", "北海道"]:
            boundsarr = df[df.is_valid].to_crs(epsg=3857).unary_union.bounds
            center = df[df.is_valid].to_crs(epsg=3857).unary_union.centroid.coords.xy

        yL = boundsarr[3] - boundsarr[1]

        if reaspect:
            boundsarr = [
                center[0][0] - yL / 2 * desired_aspect_ratio,
                boundsarr[1],
                center[0][0] + yL / 2 * desired_aspect_ratio,
                boundsarr[3],
            ]

        xL = boundsarr[2] - boundsarr[0]

        ax.set_xlim([boundsarr[0] - xBuffer * xL, boundsarr[2] + xBuffer * xL])
        ax.set_ylim([boundsarr[1] - yBuffer * yL, boundsarr[3] + yBuffer * yL])

        fig.savefig(
            PLOT_FOLDER + PREFECTURE_EN + "-profit-map_" + lang + ".pdf",
            bbox_inches="tight",
        )
        fig.savefig(
            PLOT_FOLDER + PREFECTURE_EN + "-profit-map_" + lang + ".png",
            bbox_inches="tight",
        )
        plt.close("all")

        shutil.copy(
            PLOT_FOLDER + PREFECTURE_EN + "-profit-map_" + lang + ".pdf",
            productionPlotFolder,
        )

        plt.close("all")


for PREFECTURE in ["秋田県", "北海道"]:
    print(PREFECTURE)
    make_pref(PREFECTURE)
