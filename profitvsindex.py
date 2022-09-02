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
import shutil
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from japandata.readings.data import names_df, pref_names_df

from japandata.population.data import local_pop_df, prefecture_pop_df
from japandata.furusatonouzei.data import furusato_arr, furusato_df, furusato_pref_df
from japandata.maps.data import load_map
from japandata.indices.data import local_ind_df, pref_ind_df, prefmean_ind_df

PLOT_FOLDER = os.path.join(os.getcwd(), 'profitvsindex/')
EXTRA_PLOT_FOLDER = './furusato-private/draft/figures/'

output_filetypes = ['pdf']

os.makedirs(PLOT_FOLDER, exist_ok=True)

oku = 10**8
hyakuman = 10**6

###################################
##### Merging pop data over years #
###################################
year_clone_df = local_pop_df.loc[local_pop_df['year'] == 2020].copy()
year_clone_df['year'] = 2021
local_pop_df = pd.concat([local_pop_df, year_clone_df])
year_clone_df = prefecture_pop_df.loc[prefecture_pop_df['year'] == 2020].copy()
year_clone_df['year'] = 2021
prefecture_pop_df = pd.concat([prefecture_pop_df, year_clone_df])

fn_pop_df = pd.merge(furusato_df, local_pop_df, on=[
                     "code", "year", "prefecture"], validate='one_to_one')
furusato_df.loc[~furusato_df['code'].isin(
    fn_pop_df['code']) & ~(furusato_df['city'] == 'unassigned')]
fn_pop_pref_df = pd.merge(furusato_pref_df, prefecture_pop_df, on=[
                          "year", "prefecture"], validate='one_to_one')

#######################################
###### Computing some useful things ###
#######################################
fn_pop_pref_df['profitperperson'] = fn_pop_pref_df.apply(
    lambda row: row['netgainminusdeductions']/row['total-pop'], axis=1)
totalbyyear = furusato_pref_df.groupby('year').sum()['donations']
fn_pop_pref_df['donationsfraction'] = fn_pop_pref_df.apply(
    lambda row: row['donations']/totalbyyear[row['year']], axis=1)
fn_pop_pref_df['donationsperperson'] = fn_pop_pref_df.apply(
    lambda row: row['donations']/row['total-pop'], axis=1)


############################################################################
#### PROFIT AND ECONOMIC STRENGTH: JOIN THE TWO ARRAYS  ###########################
############################################################################
year_clone_df = local_ind_df.loc[local_ind_df['year'] == 2020].copy()
year_clone_df['year'] = 2021
local_ind_df = pd.concat([local_ind_df, year_clone_df])
year_clone_df = prefmean_ind_df.loc[prefmean_ind_df['year'] == 2020].copy()
year_clone_df['year'] = 2021
prefmean_ind_df = pd.concat([prefmean_ind_df, year_clone_df])


fn_ind_df = pd.merge(fn_pop_df, local_ind_df, on=[
                     "code", "year", "prefecture"], validate='one_to_one')
## Why are the city names different? Doesn't really it matter so long as the code matches?
# fn_ind_df.loc[fn_ind_df['city_x']!=fn_ind_df['city_y']][['city_x', 'city_y']]
## Make sure only unassigned regions have no counterpart
fn_pop_df.loc[~fn_pop_df['code'].isin(fn_ind_df['code']) & ~(
    fn_pop_df['city_x'] == 'unassigned')]

tokyo23codes = [str(13101 + i) for i in range(23)]
fn_ind_df_no23 = fn_ind_df.drop(
    fn_ind_df.loc[fn_ind_df['code'].isin(tokyo23codes)].index)


fn_ind_pref_df = pd.merge(fn_pop_pref_df, prefmean_ind_df, on=[
                          "year", "prefecture"], validate='one_to_one')
#fn_ind_pref_df = pd.merge(fn_pop_pref_df, pref_ind_df, on=["year","prefecture"],validate='one_to_one')

###################################
##### Summing over years ##########
###################################

local_sum_df = fn_pop_df.groupby(['code', 'prefecturecity', 'prefecture', 'city_x']).sum(
).reset_index().drop('year', axis=1)
pref_sum_df = fn_pop_pref_df.groupby(
    ['prefecture']).sum().reset_index().drop('year', axis=1)

################
### Aliases ####
################

local_df = fn_ind_df
pref_df = fn_ind_pref_df

####################
### Map Styling ###
####################

unhighlighted_style = {
    "color": "black",
    "weight": 1,
    "fillOpacity": 1,
}

highlighted_style = unhighlighted_style | {'weight': 4}

map_style = {'location': [35.67, 139],
             'zoom_start': 5, 'tiles': 'None', 'attr': " "}

##############################
#### PROFIT VS STRENGTH -- PREF   ##
##############################
year = [2021]
titles = {'en': r"\textbf{Profits don't go to the neediest prefectures}",
                 'jp': r"\textbf{利益は困難な都道府県にいってない}"}
subtitles = {'en': r"Prefectural profit per capita (" + str(year[0])+ r") vs economic strength",
                    'jp': r"各都道府県の純利益（" + str(year[0])+"年）"+r"対財政力指標"}
#ylabels = {'en': r'Profit per capita', 'jp': r'一人当たりの純利益'}
xlabels = {'en': r'Economic Strength Index', 'jp': r'財政力指標'}

langs = ['en', 'jp']
from samplot.circusboy import CircusBoy

cb = CircusBoy(baseFont=['Helvetica','Hiragino Maru Gothic Pro'],titleFont=['Helvetica','Hiragino Maru Gothic Pro'],textFont=['Helvetica','Hiragino Maru Gothic Pro'], fontsize=12,figsize=(6,4))

for lang in langs:
    fig, ax = cb.handlers()
    cb.set_titleSubtitle(ax, titles[lang], subtitles[lang])
    if lang =='en':
        cb.set_yLabel(ax, yLabel=r'/person', currency=r'\textyen')
    if lang =='jp':
        cb.set_yLabel(ax, yLabel=r'円／人', currency=r'')   
    cb.set_yTickLabels(ax)
    
    ax.set_xlabel(xlabels[lang], color='black')

    posinds = np.where(
        fn_ind_pref_df.loc[fn_ind_pref_df['year'].isin(year)]['profitperperson'] >= 0)
    neginds = np.where(
        fn_ind_pref_df.loc[fn_ind_pref_df['year'].isin(year)]['profitperperson'] < 0)
    ax.scatter(fn_ind_pref_df.loc[fn_ind_pref_df['year'].isin(year)]['economic-strength-index'].iloc[posinds],
               fn_ind_pref_df.loc[fn_ind_pref_df['year'].isin(year)]['profitperperson'].iloc[posinds], color=samcolors.nice_colors(0))
    ax.scatter(fn_ind_pref_df.loc[fn_ind_pref_df['year'].isin(year)]['economic-strength-index'].iloc[neginds],
               fn_ind_pref_df.loc[fn_ind_pref_df['year'].isin(year)]['profitperperson'].iloc[neginds], color=samcolors.nice_colors(3))

    ax.set_xlim([.15, 1])
    ax.set_ylim([-6200, 21500])
    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.1f}'.format))
    #ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.axhline(0, color='black', alpha=.5, ls='--')

    positions = {'en':
              {"神奈川県": 'left', "大阪府": 'top', "愛知県": 'top', "埼玉県": 'top', "兵庫県": 'bottom', "岐阜県":'top', "北海道": 'right', "静岡県": 'top', "茨城県": 'top', "京都府": 'bottom', "奈良県": 'bottom', "佐賀県": 'top', "鹿児島県": 'top', "宮崎県":'bottom',
                  "山形県":'top', "熊本県": 'right', "新潟県": 'top', "徳島県": 'left', "鳥取県": 'top', "青森県": 'top', "高知県": 'bottom', "山梨県":'top', "和歌山県": 'top', '島根県':'top', '秋田県':'bottom', '福井県':'top', '群馬県':'right'},
              'jp':   {"神奈川県": 'left', "大阪府": 'top', "愛知県": 'top', "埼玉県": 'top', "兵庫県": 'bottom', "岐阜県":'top', "北海道": 'right', "静岡県": 'top', "茨城県": 'top', "京都府": 'bottom', "奈良県": 'bottom', "佐賀県": 'top', "鹿児島県": 'top', "宮崎県":'bottom',
                  "山形県":'top', "熊本県": 'right', "新潟県": 'top', "徳島県": 'left', "鳥取県": 'top', "青森県": 'top', "高知県": 'bottom', "山梨県":'top', "和歌山県": 'top', '島根県':'top', '秋田県':'bottom', '福井県':'top','群馬県':'right'}}
    # deltas = {'en':
    #           {"神奈川県": [0, 2000], "大阪府": [-.01, 0], "愛知県": [0, 2000], "埼玉県": [0, -100], "兵庫県": [0, -100], "岐阜県":[0,400], "北海道": [.05, -500], "静岡県": [0, 200], "茨城県": [0, 200], "京都府": [-0, -0], "奈良県": [-0, 0], "佐賀県": [+.03, -800], "鹿児島県": [0, .0], "宮崎県": [0, .0],
    #               "山形県": [0, -1700], "熊本県": [0, 100], "新潟県": [-.004, 0], "徳島県": [-0.05, 200], "鳥取県": [0, -1500], "青森県": [-.038, -600], "高知県": [0, 100], "山梨県": [0, 100], "和歌山県": [.055, -950], '島根県': [-.01, 100], '秋田県': [0, 100], '福井県': [0, 100], '岡山県': [.03, -100]},
    #           'jp':  {"神奈川県": [-.03, 1700], "大阪府": [-.01, -100], "愛知県": [0, 1800], "埼玉県": [0, 0], "兵庫県": [-0, 0], "福岡県": [0, 200], "北海道": [0, -1800], "静岡県": [0, 200], "茨城県": [0, 200], "京都府": [-0, -0], "奈良県": [-0, 0], "佐賀県": [+.03, -800], "鹿児島県": [0, .0], "宮崎県": [0, 200], "山形県": [0, -1800], "熊本県": [0, 100], "新潟県": [-.004, 0], "徳島県": [-0.01, 50], "鳥取県": [0, -1700], "青森県": [-.025, -900], "高知県": [0, 100], "山梨県": [0, 100], "和歌山県": [.035, -950], '島根県': [-.01, 100], '秋田県': [0, 100], '福井県': [0, 100], '岡山県': [.01, -100], '群馬県': [.025,800]}}
    for prefecture in fn_pop_pref_df['prefecture'].unique():
        if lang == 'jp':
                label = prefecture.strip("県").strip("都").strip("府")
        else:
                label = pref_names_df.loc[pref_names_df['prefecture']
                                          == prefecture, 'prefecture-reading'].iloc[0]
                label = r'\phantom{h}'+label.replace('ou', 'o').replace('oo', 'o').title()+r'\phantom{g}'
        
        x = fn_ind_pref_df.loc[(fn_ind_pref_df['year'].isin(year)) & (
            fn_ind_pref_df['prefecture'] == prefecture)]['economic-strength-index']
        y = fn_ind_pref_df.loc[(fn_ind_pref_df['year'].isin(year)) & (
            fn_ind_pref_df['prefecture'] == prefecture)]['profitperperson']
        # if prefecture in deltas[lang].keys():
        if prefecture in positions[lang].keys():
            #delta = deltas[lang][prefecture]
            position = positions[lang][prefecture]
            deltax = 0 
            deltay = 0
            if lang == 'jp':
                ypad = 600
                xpad = .01
            elif lang == 'en':
                ypad = 400
                xpad = 0
            if position == 'left':
                va = 'center'
                ha = 'right'
                deltax -= xpad
            elif position == 'right':
                va = 'center'
                ha = 'left'
                deltax += xpad
            elif position == 'top':
                va = 'bottom'
                ha = 'center'
                deltay += ypad
            elif position == 'bottom':
                va = 'top'
                ha = 'center'
                deltay -= ypad
            ax.text(x+deltax, y+deltay, label,
                    fontsize=8, ha=ha, va=va, color=cb.grey)

    #np.mean(fn_ind_pref_df.loc[(fn_ind_pref_df['year'].isin(year)) & (fn_ind_pref_df['prefecture']!='東京都')]['economic-strength-index'])
    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+'profitperperson_vs_strength_' +
                    lang+'.'+suffix, transparent=True, bbox_inches='tight')
    plt.close('all')
    shutil.copy(PLOT_FOLDER+'profitperperson_vs_strength_' +
                lang+'.pdf', EXTRA_PLOT_FOLDER)


fig = px.scatter(fn_ind_pref_df.loc[fn_ind_pref_df['year'].isin(
    year)], x='economic-strength-index', y='profitperperson', color='prefecture', hover_data=['prefecture'])
fig.write_html(PLOT_FOLDER+'/profitperperson_vs_strength_pref.html')
plt.close('all')
