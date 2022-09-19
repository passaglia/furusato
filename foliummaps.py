import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import folium
from folium.features import GeoJsonTooltip
from samplot.utils import init_plotting
from samplot.baseplot import BasePlot
from samplot.circusboy import CircusBoy

import samplot.colors as samcolors
import branca.colormap as cm
import plotly.express as px
import plotly.graph_objects as go
import shutil

from data import local_map_df, pref_map_df

PLOT_FOLDER = os.path.join(os.getcwd(),'foliummaps/')
EXTRA_PLOT_FOLDER = './furusato-private/draft/figures/'

output_filetypes = ['pdf']

os.makedirs(PLOT_FOLDER, exist_ok=True)

oku = 10**8
hyakuman = 10**6

################################
#### Restricting to one year ###
#################################
year = 2021
local_map_df_year = local_map_df.loc[local_map_df['year']==year]
pref_map_df_year = pref_map_df.loc[pref_map_df['year']==year]

####################
### Map Styling ###
####################

unhighlighted_style = {
"color": "black",
"weight": 1,
"fillOpacity": 1,
}

highlighted_style = unhighlighted_style | {'weight':4}

center =  local_map_df_year[local_map_df_year.is_valid].unary_union.centroid.coords.xy
#center = [[35.67], [139]]
map_style = {'location':[center[1][0], center[0][0]], 'zoom_start':5, 'tiles':'None','attr':" "}

def fillColor(colormap, feature):
    try: 
        return colormap(feature["properties"]['dummy']) 
    except ValueError: 
        print(feature["properties"]["name"])
        return 'black'
    except KeyError:
        print(feature["properties"]["name"])
        return 'black'

tooltipstyle = """
        background-color: #F0EFEF;
        border: 2px solid black;
        border-radius: 3px;
        box-shadow: 3px;
    """

##############################
#### FOLIUM MAP OF DONATIONS #
##############################

datacolumn= "donations"
datacolumnalias = "Total Donations (百万円)"
scalingfactor = hyakuman

df = local_map_df_year.copy()
df['dummy'] = df[datacolumn]/scalingfactor

largestdeviation = np.max(np.abs(df['dummy']))

rgbacolors = (matplotlib.cm.get_cmap('coolwarm')(np.linspace(0.5,1,11)))
colormap = cm.LinearColormap( [matplotlib.colors.to_hex(color) for color in rgbacolors],
    vmin=0, vmax=largestdeviation
)

m = folium.Map(**map_style)

tooltip = GeoJsonTooltip(
    fields=["prefecture", "city", "dummy"],
    aliases=["Prefecture:", "City:", datacolumnalias + ':'],
    localize=True,
    sticky=False,
    labels=True,
    style=tooltipstyle,
    max_width=800,
)

folium.GeoJson(df
,name = 'dummy'
,style_function=lambda feature: 
{'fillColor':fillColor(colormap, feature)} | unhighlighted_style
,highlight_function=lambda feature: 
{'fillColor':fillColor(colormap, feature)} | highlighted_style
,zoom_on_click=True
,tooltip=tooltip
).add_to(m)

colormap.caption = datacolumnalias
colormap.add_to(m)

m.save(PLOT_FOLDER+"donations_map.html")

##############################
#### FOLIUM MAP OF PROFIT: PREFECTURE ###
##############################

datacolumn= "netgainminusdeductions"
datacolumnalias = "Net Gain Minus Deductions (百万円)"
scalingfactor = hyakuman

df = pref_map_df_year.copy()
df['dummy'] = df[datacolumn]/scalingfactor

largestdeviation = 20000

rgbacolors = (matplotlib.cm.get_cmap('coolwarm')(np.linspace(0,1,11)))
colormap = cm.LinearColormap( [matplotlib.colors.to_hex(color) for color in rgbacolors],
    vmin=-largestdeviation, vmax=largestdeviation
)

m = folium.Map(**map_style)

tooltip = GeoJsonTooltip(
    fields=["prefecture", "dummy"],
    aliases=["Prefecture:", datacolumnalias + ':'],
    localize=True,
    sticky=False,
    labels=True,
    style=tooltipstyle,
    max_width=800,
)

folium.GeoJson(df
,name = 'dummy'
,style_function=lambda feature: 
{'fillColor':fillColor(colormap, feature)} | unhighlighted_style
,highlight_function=lambda feature: 
{'fillColor':fillColor(colormap, feature)} | highlighted_style
,zoom_on_click=True
,tooltip=tooltip
).add_to(m)

colormap.caption = datacolumnalias
try:
    colormap.add_to(m)
except AttributeError:
    pass

m.save(PLOT_FOLDER+"profit_prefecture_map.html")

##############################
#### FOLIUM MAP OF PROFIT ###
##############################

datacolumn= "netgainminusdeductions"
datacolumnalias = "Net Gain Minus Deductions (百万円)"
scalingfactor = hyakuman
df = local_map_df_year.copy()
df['dummy'] = df[datacolumn]/scalingfactor

#largestdeviation = np.max(np.abs(map_df[datacolumn]))
largestdeviation = 2000

rgbacolors = (matplotlib.cm.get_cmap('coolwarm')(np.linspace(0,1,11)))
colormap = cm.LinearColormap( [matplotlib.colors.to_hex(color) for color in rgbacolors],
    vmin=-largestdeviation, vmax=largestdeviation
)

m = folium.Map(**map_style)

tooltip = GeoJsonTooltip(
    fields=["prefecture", "city", 'dummy'],
    aliases=["Prefecture:", "City:", datacolumnalias + ':'],
    localize=True,
    sticky=False,
    labels=True,
    style=tooltipstyle,
    max_width=800,
)

folium.GeoJson(df
,name = 'dummy'
,style_function=lambda feature: 
{'fillColor':fillColor(colormap, feature)} | unhighlighted_style
,highlight_function=lambda feature: 
{'fillColor':fillColor(colormap, feature)} | highlighted_style
,zoom_on_click=True
,tooltip=tooltip
).add_to(m)

colormap.caption = datacolumnalias
colormap.add_to(m)

m.save(PLOT_FOLDER+"profit_map.html")


####################################################################
#### PER PERSON PROFIT-LOSS MAP ####################################
####################################################################
datacolumn= "profit-per-person"
datacolumnalias = "Profit per person (円/person)"
scalingfactor = 1

df = pref_map_df_year.copy()
df['dummy'] = df[datacolumn]/scalingfactor

#largestdeviation = np.max(np.abs(map_df[datacolumn]))
largestdeviation = 20000

rgbacolors = (matplotlib.cm.get_cmap('coolwarm')(np.linspace(0,1,11)))
colormap = cm.LinearColormap( [matplotlib.colors.to_hex(color) for color in rgbacolors],
    vmin=-largestdeviation, vmax=largestdeviation)
colormap = lambda x: matplotlib.colors.to_hex(matplotlib.cm.get_cmap('coolwarm')(matplotlib.colors.TwoSlopeNorm(0, vmin=-5000, vmax=20000)(x)))
m = folium.Map(**map_style)

tooltip = GeoJsonTooltip(
    fields=["prefecture", datacolumn],
    aliases=["Prefecture:", datacolumnalias + ':'],
    localize=True,
    sticky=False,
    labels=True,
    style=tooltipstyle,
    max_width=800,
)

folium.GeoJson(df
,name = 'dummy'
,style_function=lambda feature: 
{'fillColor':fillColor(colormap, feature)} | unhighlighted_style
,highlight_function=lambda feature: 
{'fillColor':fillColor(colormap, feature)} | highlighted_style
,zoom_on_click=True
,tooltip=tooltip
).add_to(m)

colormap.caption = datacolumnalias
try:
    colormap.add_to(m)
except AttributeError:
    pass

m.save(PLOT_FOLDER+"profitperperson_prefecture_map.html")

######################################
### ORDERED PER PERSON PROFIT-LOSS  ##
######################################

# year = 2021

# titles = {'en':r"\textbf{Winners and losers}", 
# 'jp':r"\textbf{都道府県の間に、利益格差が広がっている}"}
# subtitles = {'en':r"Ranked profit per capita ", 
# 'jp':str(year) + r"各県の人口と一人当たり純利益の進化"}

# ylabels = {'en':r'Profit per capita (\textyen/person)', 
# 'jp':r'人口 (百万人）'} 
# langs = ['en','jp']

# datacolumn = 'profitperperson'
# #datacolumn = 'netgainminusdeductions'

# for lang in langs: 
#     fig, ax = init_plotting(style='nyt')
#     ax.set_title(subtitles[lang], x=0., y=1.05, fontsize=14,ha='left',va='bottom',wrap=True)
#     fig.suptitle(titles[lang], x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes, wrap=True)
#     ax.set_axisbelow(True)
#     ax.set_ylabel(ylabels[lang])
#     ax.get_xaxis().set_visible(False)
#     ax.spines['bottom'].set_visible(False)

#     #ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
#     ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    
#     winners = fn_pop_pref_df.loc[(fn_pop_pref_df['year']==year) & (fn_pop_pref_df[datacolumn]>0)].sort_values(datacolumn,ascending=False)
#     losers = fn_pop_pref_df.loc[(fn_pop_pref_df['year']==year) & (fn_pop_pref_df[datacolumn]<0)].sort_values(datacolumn,ascending=False)

#     barwin = ax.bar(np.arange(len(winners)),winners[datacolumn], color=samcolors.nice_colors(0))
#     barlose = ax.bar(np.arange(len(winners),len(winners)+len(losers)),losers[datacolumn], color=samcolors.nice_colors(3))

#     ax.bar_label(barwin, winners['prefecture'],padding=1, rotation=10,ha='left', va='bottom')

#     # for prefecture in fn_pop_pref_df['prefecture'].unique(): 

#     #     xs = fn_pop_pref_df.loc[(fn_pop_pref_df['prefecture']==prefecture) &(fn_pop_pref_df['year'].isin(year))]['profitperperson'].values
#     #     ys = fn_pop_pref_df.loc[(fn_pop_pref_df['prefecture']==prefecture) & (fn_pop_pref_df['year'].isin(year))]['total-pop'].values / 10**6
#     #     #ax.plot(fn_pop_pref_df.loc[(fn_pop_pref_df['prefecture']==prefecture) &(fn_pop_pref_df['year'].isin(year))]['profitperperson'],fn_pop_pref_df.loc[(fn_pop_pref_df['prefecture']==prefecture) & (fn_pop_pref_df['year'].isin(year))]['total-pop'])
#     #     if xs[1]<0:
#     #         color = samcolors.nice_colors(3)
#     #     else: 
#     #         color = samcolors.nice_colors(0)
#     #     ax.annotate("", xy=(xs[0], ys[0]), xytext=(xs[1], ys[1]),
#     #             arrowprops=dict(arrowstyle="<-",shrinkA=0, shrinkB=0,color=color),fontsize=10)

#     #     if lang == 'jp':
#     #         deltas = deltas_jp
#     #     else:
#     #         deltas = deltas_en
#     #     try: delta = deltas[prefecture]
#     #     except KeyError: delta = [0,0]

#     #     if prefecture in deltas.keys():
#     #         if lang == 'jp':
#     #             label = prefecture.strip("県").strip("都").strip("府")
#     #         else:
#     #             label = pref_names_df.loc[pref_names_df['prefecture']==prefecture, 'prefecture-reading'].iloc[0]
#     #             label = label.replace('ou','o').replace('oo','o').title()
#     #         ax.text(np.mean(xs)+delta[0],np.mean(ys)+delta[1],label, fontsize=8, ha='center',va='center')
        
#     #     if prefecture == '東京都':
#     #         ax.text(xs[0],ys[0]+.15,year[0], fontsize=8, ha='center',va='bottom')
#     #         ax.text(xs[1]+300,ys[1]-.2,year[1], fontsize=8, ha='center',va='top')

#     # ax.set_ylim([0,1.41*10])
#     for suffix in output_filetypes:
#         fig.savefig(PLOT_FOLDER+'profitperperson_ordered_'+lang+'.'+suffix, transparent=True,bbox_inches="tight")
#     plt.close('all')

# shutil.copy(PLOT_FOLDER+'profitperperson_ordered_en.pdf',EXTRA_PLOT_FOLDER)
# shutil.copy(PLOT_FOLDER+'profitperperson_ordered_jp.pdf',EXTRA_PLOT_FOLDER)
