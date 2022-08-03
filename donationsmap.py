import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import folium
from folium.features import GeoJsonTooltip
from samplot.utils import init_plotting
import samplot.colors as samcolors
import seaborn as sns
import branca.colormap as cm
import plotly.express as px
import plotly.graph_objects as go

from japandata.population.data import local_pop_df, prefecture_pop_df
from japandata.furusatonouzei.data import furusato_arr, furusato_df, furusato_pref_df
from japandata.maps.data import load_map
from japandata.indices.data import local_ind_df, pref_ind_df, prefmean_ind_df

PLOT_FOLDER = os.path.join(os.getcwd(),'advancedgraphics/')
MAP_FOLDER = os.path.join(PLOT_FOLDER,'maps/')
output_filetypes = ['pdf']

os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(MAP_FOLDER, exist_ok=True)

oku = 10**8
hyakuman = 10**6

###################################
##### Merging pop data over years ##########
###################################
fn_pop_df = pd.merge(furusato_df, local_pop_df, on=["code", "year", "prefecture"],validate='one_to_one')
furusato_df.loc[~furusato_df['code'].isin(fn_pop_df['code']) & ~(furusato_df['city']=='unassigned')]
fn_pop_pref_df = pd.merge(furusato_pref_df, prefecture_pop_df, on=["year","prefecture"],validate='one_to_one')

############################################################################
#### Merging with economic data  ###########################
############################################################################

fn_pop_ind_df = pd.merge(fn_pop_df, local_ind_df, on=["code", "year", "prefecture"],validate='one_to_one')

## Make sure only unassigned regions have no counterpart
#fn_pop_df.loc[~fn_pop_df['code'].isin(fn_ind_df['code']) & ~(fn_pop_df['city_x']=='unassigned')]

tokyo23codes = [str(13101 + i) for i in range(23)]
fn_pop_ind_df_no23 = fn_pop_ind_df.drop(fn_pop_ind_df.loc[fn_pop_ind_df['code'].isin(tokyo23codes)].index)


fn_pop_ind_pref_df = pd.merge(fn_pop_pref_df, prefmean_ind_df, on=["year","prefecture"],validate='one_to_one')
#fn_ind_pref_df = pd.merge(fn_pop_pref_df, pref_ind_df, on=["year","prefecture"],validate='one_to_one')

##################################
#### Summing over years ##########
##################################

local_sum_df = fn_pop_ind_df.groupby(['code','prefecturecity','prefecture', 'city_x']).sum().reset_index().drop('year', axis=1)
pref_sum_df = fn_pop_ind_pref_df.groupby(['prefecture']).sum().reset_index().drop('year', axis=1)

######################################
### Aliasing ####
######################################

local_df = fn_pop_ind_df
pref_df = fn_pop_ind_pref_df

#######################################
###### Computing some useful things ###
#######################################
pd.options.mode.chained_assignment = None
local_df['profit-per-person'] = local_df.apply(lambda row: row['netgainminusdeductions']/row['total-pop'],axis=1)
totalbyyear = local_df.groupby('year').sum()['donations']
local_df['donations-fraction'] = local_df.apply(lambda row: row['donations']/totalbyyear[row['year']],axis=1)
local_df['donations-per-person'] = local_df.apply(lambda row: row['donations']/row['total-pop'],axis=1)

pref_df['profit-per-person'] = pref_df.apply(lambda row: row['netgainminusdeductions']/row['total-pop'],axis=1)
totalbyyear = pref_df.groupby('year').sum()['donations']
pref_df['donations-fraction'] = pref_df.apply(lambda row: row['donations']/totalbyyear[row['year']],axis=1)
pref_df['donations-per-person'] = pref_df.apply(lambda row: row['donations']/row['total-pop'],axis=1)

############################################
#### Restricting to one year and cleanup ###
############################################
year = 2020
local_df_year = local_df.loc[local_df['year']==year]
local_df_year = local_df_year.drop(['city','city_x','city_y'],axis=1)
local_df_year= local_df_year.reset_index(drop=True)

pref_df_year = pref_df.loc[pref_df['year']==year]

##############################################################
### Adding map  ###
##############################################################

map_df = load_map(year,level='local_dc')
pref_map_df = load_map(year,level='prefecture')

map_df = map_df.drop(map_df.loc[~map_df['code'].isin(local_df_year['code'])].index,axis=0)
map_df = map_df.reset_index(drop=True)

local_df_with_map = pd.merge(map_df, local_df_year, on=["code", "prefecture"],validate='one_to_one')
pref_df_with_map = pd.merge(pref_map_df, pref_df_year, on=["code", "prefecture"],validate='one_to_one')

####################
### Map Styling ###
####################

unhighlighted_style = {
"color": "black",
"weight": 1,
"fillOpacity": 1,
}

highlighted_style = unhighlighted_style | {'weight':4}

map_style = {'location':[35.67, 139], 'zoom_start':5, 'tiles':'None','attr':" "}

############################################
#### FOLIUM MAP OF DONATIONS: PREFECTURE ###
############################################

# #df = furusato_pref_df.loc[furusato_pref_df['year']==mapyear]
# df = furusato_pref_sum_df

# datacolumn= "donations"
# datacolumnalias = "Total Donations (100 Million Yen)"
# scalingfactor = oku

# map_df[datacolumn] = 0
# for i in range(len(map_df)):
#     try: 
#         map_df.loc[i, datacolumn] =  (df.loc[df['prefecture'] == map_df.loc[i,'prefecture'],datacolumn].values[0]/scalingfactor)
#     except IndexError: 
#         print(map_df.loc[i,'code'])
#         print(map_df.loc[i,'city'])

# largestdeviation = np.max(np.abs(map_df[datacolumn]))
# #largestdeviation = 10000

# rgbacolors = (matplotlib.cm.get_cmap('coolwarm')(np.linspace(0.5,1,11)))
# colormap = cm.LinearColormap( [matplotlib.colors.to_hex(color) for color in rgbacolors],
#     vmin=0, vmax=largestdeviation
# )

# m = folium.Map(**map_style)

# tooltip = GeoJsonTooltip(
#     fields=["prefecture", datacolumn],
#     aliases=["Prefecture:", datacolumnalias + ':'],
#     localize=True,
#     sticky=False,
#     labels=True,
#     style="""
#         background-color: #F0EFEF;
#         border: 2px solid black;
#         border-radius: 3px;
#         box-shadow: 3px;
#     """,
#     max_width=800,
# )

# def fillColor(feature):
#     try: 
#         return colormap(feature["properties"][datacolumn]) 
#     except ValueError: 
#         print(feature["properties"]["name"])
#         return 'black'
#     except KeyError:
#         print(feature["properties"]["name"])
#         return 'black'

# folium.GeoJson(map_df
# ,name = datacolumn
# ,style_function=lambda feature: 
# {'fillColor':fillColor(feature)} | unhighlighted_style
# ,highlight_function=lambda feature: 
# {'fillColor':fillColor(feature)} | highlighted_style
# ,zoom_on_click=True
# ,tooltip=tooltip
# ).add_to(m)

# colormap.caption = datacolumnalias
# colormap.add_to(m)

# m.save(MAP_FOLDER+"donations_prefecture_map.html")




# ##############################
# #### FOLIUM MAP OF DONATIONS ###
# ##############################

# mapyear = 2020
# map_df = load_map(mapyear,level='local_dc')
# #df = furusato_df.loc[furusato_df['year']==mapyear]
# df = furusato_sum_df

# datacolumn= "donations"
# datacolumnalias = "Total Donations (百万円)"
# scalingfactor = hyakuman

# map_df[datacolumn] = 0
# for i in range(len(map_df)):
#     try: 
#         map_df.loc[i, datacolumn] =  (df.loc[df['code'] == map_df.loc[i,'code'],datacolumn].values[0]/scalingfactor)
#     except IndexError: 
#         print(map_df.loc[i,'code'])
#         print(map_df.loc[i,'city'])

# map_df.loc[map_df['city'] ==  '双葉町'].code
# df.loc[df['city'] == '双葉町'].code
# #largestdeviation = np.max(np.abs(map_df[datacolumn]))
# largestdeviation = 1000

# rgbacolors = (matplotlib.cm.get_cmap('coolwarm')(np.linspace(0.5,1,11)))
# colormap = cm.LinearColormap( [matplotlib.colors.to_hex(color) for color in rgbacolors],
#     vmin=0, vmax=largestdeviation
# )

# m = folium.Map(**map_style)

# tooltip = GeoJsonTooltip(
#     fields=["prefecture", "city", datacolumn],
#     aliases=["Prefecture:", "City:", datacolumnalias + ':'],
#     localize=True,
#     sticky=False,
#     labels=True,
#     style="""
#         background-color: #F0EFEF;
#         border: 2px solid black;
#         border-radius: 3px;
#         box-shadow: 3px;
#     """,
#     max_width=800,
# )

# def fillColor(feature):
#     try: 
#         return colormap(feature["properties"][datacolumn]) 
#     except ValueError: 
#         print(feature["properties"]["name"])
#         return 'black'
#     except KeyError:
#         print(feature["properties"]["name"])
#         return 'black'

# folium.GeoJson(map_df
# ,name = datacolumn
# ,style_function=lambda feature: 
# {'fillColor':fillColor(feature)} | unhighlighted_style
# ,highlight_function=lambda feature: 
# {'fillColor':fillColor(feature)} | highlighted_style
# ,zoom_on_click=True
# ,tooltip=tooltip
# ).add_to(m)

# colormap.caption = datacolumnalias
# colormap.add_to(m)

# m.save(MAP_FOLDER+"donations_map.html")

##############################
#### FOLIUM MAP OF DONATIONS #
##############################

datacolumn= "donations"
datacolumnalias = "Total Donations (百万円)"
scalingfactor = hyakuman

df = local_df_with_map.copy()
df['dummy'] = df[datacolumn]/scalingfactor

largestdeviation = np.max(np.abs(df['dummy']))
#largestdeviation = 1000

rgbacolors = (matplotlib.cm.get_cmap('coolwarm')(np.linspace(0.5,1,11)))
colormap = cm.LinearColormap( [matplotlib.colors.to_hex(color) for color in rgbacolors],
    vmin=0, vmax=largestdeviation
)

center =  df[df.is_valid].unary_union.centroid.coords.xy
m = folium.Map(**{'location':[center[1][0], center[0][0]], 'zoom_start':9, 'tiles':'None','attr':" "})

tooltip = GeoJsonTooltip(
    fields=["prefecture", "city", "dummy"],
    aliases=["Prefecture:", "City:", datacolumnalias + ':'],
    localize=True,
    sticky=False,
    labels=True,
    style="""
        background-color: #F0EFEF;
        border: 2px solid black;
        border-radius: 3px;
        box-shadow: 3px;
    """,
    max_width=800,
)

def fillColor(feature):
    try: 
        return colormap(feature["properties"]['dummy']) 
    except ValueError: 
        print(feature["properties"]["name"])
        return 'black'
    except KeyError:
        print(feature["properties"]["name"])
        return 'black'

folium.GeoJson(df
,name = 'dummy'
,style_function=lambda feature: 
{'fillColor':fillColor(feature)} | unhighlighted_style
,highlight_function=lambda feature: 
{'fillColor':fillColor(feature)} | highlighted_style
,zoom_on_click=True
,tooltip=tooltip
).add_to(m)

colormap.caption = datacolumnalias
colormap.add_to(m)

m.save(PLOT_FOLDER+PREFECTURE+"donations_map.html")