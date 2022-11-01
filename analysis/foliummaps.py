import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import folium
from folium.features import GeoJsonTooltip

import branca.colormap as cm
import shutil

from data import local_df, pref_df
from japandata.maps.data import add_df_to_map

PLOT_FOLDER = os.path.join(os.getcwd(), "foliummaps/")

output_filetypes = ["pdf"]

os.makedirs(PLOT_FOLDER, exist_ok=True)

oku = 10**8
hyakuman = 10**6

################################
#### Loading maps ###
#################################
year = 2021
pref_df = pref_df.loc[pref_df["year"] == year]
local_df = local_df.loc[local_df["year"] == year]


pref_map_df = add_df_to_map(pref_df, date=year, level="prefecture", quality="stylized")

local_map_df = add_df_to_map(local_df, date=year, level="local_dc", quality="stylized")

####################
### Map Styling ###
####################

unhighlighted_style = {
    "color": "black",
    "weight": 1,
    "fillOpacity": 1,
}

highlighted_style = unhighlighted_style | {"weight": 4}


def fillColor(colormap, feature):
    try:
        return colormap(feature["properties"]["_dummy"])
    except ValueError:
        print(feature["properties"]["name"])
        return "black"
    except KeyError:
        print(feature["properties"]["name"])
        return "black"


tooltipstyle = """
        background-color: #F0EFEF;
        border: 2px solid black;
        border-radius: 3px;
        box-shadow: 3px;
    """

center = local_map_df[local_map_df.is_valid].unary_union.centroid.coords.xy
# center = [[35.67], [139]]
map_style = {
    "location": [center[1][0], center[0][0]],
    "zoom_start": 5,
    "tiles": "None",
    "attr": " ",
}

##############################
#### FOLIUM MAP OF DONATIONS #
##############################

datacolumn = "donations"
datacolumnalias = "Total Donations (百万円)"
scalingfactor = hyakuman

df = local_map_df.copy()
df["_dummy"] = df[datacolumn] / scalingfactor

largestdeviation = np.max(np.abs(df["_dummy"]))

rgbacolors = matplotlib.cm.get_cmap("coolwarm")(np.linspace(0.5, 1, 11))
colormap = cm.LinearColormap(
    [matplotlib.colors.to_hex(color) for color in rgbacolors],
    vmin=0,
    vmax=largestdeviation,
)

m = folium.Map(**map_style)

tooltip = GeoJsonTooltip(
    fields=["prefecture", "city", "_dummy"],
    aliases=["Prefecture:", "City:", datacolumnalias + ":"],
    localize=True,
    sticky=False,
    labels=True,
    style=tooltipstyle,
    max_width=800,
)

folium.GeoJson(
    df,
    name="_dummy",
    style_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
    | unhighlighted_style,
    highlight_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
    | highlighted_style,
    zoom_on_click=True,
    tooltip=tooltip,
).add_to(m)

colormap.caption = datacolumnalias
colormap.add_to(m)

m.save(PLOT_FOLDER + "donations_map.html")

##############################
#### FOLIUM MAP OF PROFIT: PREFECTURE ###
##############################

datacolumn = "netgainminusdeductions"
datacolumnalias = "Net Gain Minus Deductions (百万円)"
scalingfactor = hyakuman

df = pref_map_df.copy()
df["_dummy"] = df[datacolumn] / scalingfactor

largestdeviation = 20000

rgbacolors = matplotlib.cm.get_cmap("coolwarm")(np.linspace(0, 1, 11))
colormap = cm.LinearColormap(
    [matplotlib.colors.to_hex(color) for color in rgbacolors],
    vmin=-largestdeviation,
    vmax=largestdeviation,
)

m = folium.Map(**map_style)

tooltip = GeoJsonTooltip(
    fields=["prefecture", "_dummy"],
    aliases=["Prefecture:", datacolumnalias + ":"],
    localize=True,
    sticky=False,
    labels=True,
    style=tooltipstyle,
    max_width=800,
)

folium.GeoJson(
    df,
    name="_dummy",
    style_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
    | unhighlighted_style,
    highlight_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
    | highlighted_style,
    zoom_on_click=True,
    tooltip=tooltip,
).add_to(m)

colormap.caption = datacolumnalias
try:
    colormap.add_to(m)
except AttributeError:
    pass

m.save(PLOT_FOLDER + "profit_prefecture_map.html")

##############################
#### FOLIUM MAP OF PROFIT ###
##############################

datacolumn = "netgainminusdeductions"
datacolumnalias = "Net Gain Minus Deductions (百万円)"
scalingfactor = hyakuman
df = local_map_df.copy()
df["_dummy"] = df[datacolumn] / scalingfactor

# largestdeviation = np.max(np.abs(map_df[datacolumn]))
largestdeviation = 2000

rgbacolors = matplotlib.cm.get_cmap("coolwarm")(np.linspace(0, 1, 11))
colormap = cm.LinearColormap(
    [matplotlib.colors.to_hex(color) for color in rgbacolors],
    vmin=-largestdeviation,
    vmax=largestdeviation,
)

m = folium.Map(**map_style)

tooltip = GeoJsonTooltip(
    fields=["prefecture", "city", "_dummy"],
    aliases=["Prefecture:", "City:", datacolumnalias + ":"],
    localize=True,
    sticky=False,
    labels=True,
    style=tooltipstyle,
    max_width=800,
)

folium.GeoJson(
    df,
    name="_dummy",
    style_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
    | unhighlighted_style,
    highlight_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
    | highlighted_style,
    zoom_on_click=True,
    tooltip=tooltip,
).add_to(m)

colormap.caption = datacolumnalias
colormap.add_to(m)

m.save(PLOT_FOLDER + "profit_map.html")


####################################################################
#### PER PERSON PROFIT-LOSS MAP ####################################
####################################################################
datacolumn = "profit-per-person"
datacolumnalias = "Profit per person (円/person)"
scalingfactor = 1

df = pref_map_df.copy()
df["_dummy"] = df[datacolumn] / scalingfactor

# largestdeviation = np.max(np.abs(map_df[datacolumn]))
largestdeviation = 20000

rgbacolors = matplotlib.cm.get_cmap("coolwarm")(np.linspace(0, 1, 11))
colormap = cm.LinearColormap(
    [matplotlib.colors.to_hex(color) for color in rgbacolors],
    vmin=-largestdeviation,
    vmax=largestdeviation,
)
colormap = lambda x: matplotlib.colors.to_hex(
    matplotlib.cm.get_cmap("coolwarm")(
        matplotlib.colors.TwoSlopeNorm(0, vmin=-5000, vmax=20000)(x)
    )
)
m = folium.Map(**map_style)

tooltip = GeoJsonTooltip(
    fields=["prefecture", datacolumn],
    aliases=["Prefecture:", datacolumnalias + ":"],
    localize=True,
    sticky=False,
    labels=True,
    style=tooltipstyle,
    max_width=800,
)

folium.GeoJson(
    df,
    name="_dummy",
    style_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
    | unhighlighted_style,
    highlight_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
    | highlighted_style,
    zoom_on_click=True,
    tooltip=tooltip,
).add_to(m)

colormap.caption = datacolumnalias
try:
    colormap.add_to(m)
except AttributeError:
    pass

m.save(PLOT_FOLDER + "profitperperson_prefecture_map.html")

#############################
### PREFECTURE LEVEL MAPS ###
#############################


def make_pref(PREFECTURE):

    print(PREFECTURE)
    PLOT_FOLDER = os.path.join(os.getcwd(), "singlepref/" + PREFECTURE + "/")
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    #### Restrict to one prefecture

    df = local_map_df.copy()
    df = df.loc[df["prefecture"] == PREFECTURE]

    ###### Computing extra field
    pd.options.mode.chained_assignment = None
    df["donations-fraction-prefecture"] = df["donations"] / df["donations"].sum()

    ###### Extra styling
    center = df[df.is_valid].unary_union.centroid.coords.xy
    # center = [[35.67], [139]]
    map_style = {
        "location": [center[1][0], center[0][0]],
        "zoom_start": 9,
        "tiles": "None",
        "attr": " ",
    }

    #### Donations map

    datacolumn = "donations"
    datacolumnalias = "Total Donations (百万円)"
    scalingfactor = hyakuman

    df["_dummy"] = df[datacolumn] / scalingfactor

    largestdeviation = np.max(np.abs(df["_dummy"]))
    # largestdeviation = 1000

    rgbacolors = matplotlib.cm.get_cmap("coolwarm")(np.linspace(0.5, 1, 11))
    colormap = cm.LinearColormap(
        [matplotlib.colors.to_hex(color) for color in rgbacolors],
        vmin=0,
        vmax=largestdeviation,
    )

    m = folium.Map(**map_style)

    tooltip = GeoJsonTooltip(
        fields=[
            "prefecture",
            "city",
            "city-reading",
            "code",
            "total-pop",
            "netgainminusdeductions",
            "donations-fraction-prefecture",
            "economic-strength-index",
            "profit-incl-ckz",
            "_dummy",
        ],
        aliases=[
            "Prefecture:",
            "City:",
            "City (en):",
            "Code:",
            "Population:",
            "Net Gain Minus Deductions:",
            "Fraction of all donations in Prefecture:",
            "Economic Strength Index",
            "Profit including ckz",
            datacolumnalias + ":",
        ],
        localize=True,
        sticky=False,
        labels=True,
        style=tooltipstyle,
        max_width=800,
    )

    folium.GeoJson(
        df,
        name="_dummy",
        style_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
        | unhighlighted_style,
        highlight_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
        | highlighted_style,
        zoom_on_click=True,
        tooltip=tooltip,
    ).add_to(m)

    colormap.caption = datacolumnalias
    colormap.add_to(m)

    m.save(PLOT_FOLDER + PREFECTURE + "donations_map.html")

    ##############################
    #### FOLIUM MAP OF PROFIT ###
    ##############################

    datacolumn = "netgainminusdeductions"
    datacolumnalias = "Net Gain Minus Deductions (百万円)"
    scalingfactor = hyakuman

    df["_dummy"] = df[datacolumn] / scalingfactor

    largestdeviation = np.max(np.abs(df["_dummy"]))
    # largestdeviation = 2000

    rgbacolors = matplotlib.cm.get_cmap("coolwarm")(np.linspace(0, 1, 11))
    colormap = cm.LinearColormap(
        [matplotlib.colors.to_hex(color) for color in rgbacolors],
        vmin=-largestdeviation,
        vmax=largestdeviation,
    )

    m = folium.Map(**map_style)

    tooltip = GeoJsonTooltip(
        fields=[
            "prefecture",
            "city",
            "city-reading",
            "code",
            "total-pop",
            "donations",
            "donations-fraction-prefecture",
            "economic-strength-index",
            "profit-incl-ckz",
            "_dummy",
        ],
        aliases=[
            "Prefecture:",
            "City:",
            "City (en):",
            "Code:",
            "Population:",
            "Donations:",
            "Fraction of all donations in Prefecture:",
            "Economic Strength Index",
            "Profit Including ckz",
            datacolumnalias + ":",
        ],
        localize=True,
        sticky=False,
        labels=True,
        style=tooltipstyle,
        max_width=800,
    )

    folium.GeoJson(
        df,
        name=datacolumn,
        style_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
        | unhighlighted_style,
        highlight_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
        | highlighted_style,
        zoom_on_click=True,
        tooltip=tooltip,
    ).add_to(m)

    colormap.caption = datacolumnalias
    colormap.add_to(m)

    m.save(PLOT_FOLDER + PREFECTURE + "profit_map.html")

    ##############################
    #### FOLIUM MAP OF PROFIT CKZ ###
    ##############################

    datacolumn = "profit-incl-ckz"
    datacolumnalias = "Profit including ckz (百万円)"
    scalingfactor = hyakuman

    df["_dummy"] = df[datacolumn] / scalingfactor

    largestdeviation = np.max(np.abs(df["_dummy"]))
    # largestdeviation = 2000

    rgbacolors = matplotlib.cm.get_cmap("coolwarm")(np.linspace(0, 1, 11))
    colormap = cm.LinearColormap(
        [matplotlib.colors.to_hex(color) for color in rgbacolors],
        vmin=-largestdeviation,
        vmax=largestdeviation,
    )
    colormap = (
        lambda val: matplotlib.colors.to_hex(rgbacolors[0])
        if val < 0
        else matplotlib.colors.to_hex(rgbacolors[-1])
    )

    m = folium.Map(**map_style)

    tooltip = GeoJsonTooltip(
        fields=[
            "prefecture",
            "city",
            "city-reading",
            "code",
            "total-pop",
            "donations",
            "donations-fraction-prefecture",
            "economic-strength-index",
            "netgainminusdeductions",
            "_dummy",
        ],
        aliases=[
            "Prefecture:",
            "City:",
            "City (en):",
            "Code:",
            "Population:",
            "Donations:",
            "Fraction of all donations in Prefecture:",
            "Economic Strength Index",
            "Net Gain Minus Deductions",
            datacolumnalias + ":",
        ],
        localize=True,
        sticky=False,
        labels=True,
        style=tooltipstyle,
        max_width=800,
    )

    folium.GeoJson(
        df,
        name=datacolumn,
        style_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
        | unhighlighted_style,
        highlight_function=lambda feature: {"fillColor": fillColor(colormap, feature)}
        | highlighted_style,
        zoom_on_click=True,
        tooltip=tooltip,
    ).add_to(m)

    # colormap.caption = datacolumnalias
    try:
        colormap.add_to(m)
    except AttributeError:
        pass

    m.save(PLOT_FOLDER + PREFECTURE + "profit_ckz_map.html")

    print(len(df.loc[df["profit-incl-ckz"] < 0]))
    print(len(df.loc[df["netgainminusdeductions"] < 0]))


print(len(local_map_df.loc[local_map_df["profit-incl-ckz"] < 0]))
print(len(local_map_df.loc[local_map_df["netgainminusdeductions"] < 0]))
# for PREFECTURE in ['北海道','秋田県']:
#     make_pref(PREFECTURE)
PREFECTURE = "北海道"
for PREFECTURE in pref_map_df["prefecture"].unique():
    print(PREFECTURE)
    make_pref(PREFECTURE)
