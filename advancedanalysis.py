import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import folium
from folium.features import GeoJsonTooltip
from samplot.circusboy import CircusBoy
import samplot.colors as samcolors

import seaborn as sns
import branca.colormap as cm
import plotly.express as px
import plotly.graph_objects as go
import shutil
from matplotlib.ticker import FuncFormatter,StrMethodFormatter

from japandata.population.data import local_pop_df, prefecture_pop_df
from japandata.furusatonouzei.data import furusato_df, furusato_pref_df, furusato_rough_df
from japandata.maps.data import load_map
from japandata.indices.data import local_ind_df, pref_ind_df, prefmean_ind_df

PLOT_FOLDER = os.path.join(os.getcwd(),'advancedgraphics/')
EXTRA_PLOT_FOLDER = './furusato-private/draft/figures/'

MAP_FOLDER = os.path.join(PLOT_FOLDER,'maps/')
output_filetypes = ['pdf']

os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(MAP_FOLDER, exist_ok=True)

oku = 10**8
hyakuman = 10**6

cb = CircusBoy(baseFont=['Helvetica','Hiragino Maru Gothic Pro'],titleFont=['Helvetica','Hiragino Maru Gothic Pro'],textFont=['Helvetica','Hiragino Maru Gothic Pro'], fontsize=12,figsize=(6,4))

###################################
##### Merging pop data over years #
###################################
year_clone_df = local_pop_df.loc[local_pop_df['year'] == 2020].copy()
year_clone_df['year'] = 2021
local_pop_df = pd.concat([local_pop_df, year_clone_df])
year_clone_df = prefecture_pop_df.loc[prefecture_pop_df['year'] == 2020].copy()
year_clone_df['year'] = 2021
prefecture_pop_df = pd.concat([prefecture_pop_df, year_clone_df])

fn_pop_df = pd.merge(furusato_df, local_pop_df, on=["code", "year", "prefecture"],validate='one_to_one')
furusato_df.loc[~furusato_df['code'].isin(fn_pop_df['code']) & ~(furusato_df['city']=='prefecture')]
fn_pop_pref_df = pd.merge(furusato_pref_df, prefecture_pop_df, on=["year","prefecture"],validate='one_to_one')

############################################################################
#### Merging with economic data  ###########################
############################################################################
year_clone_df = local_ind_df.loc[local_ind_df['year'] == 2020].copy()
year_clone_df['year'] = 2021
local_ind_df = pd.concat([local_ind_df, year_clone_df])
year_clone_df = prefmean_ind_df.loc[prefmean_ind_df['year'] == 2020].copy()
year_clone_df['year'] = 2021
prefmean_ind_df = pd.concat([prefmean_ind_df, year_clone_df])

fn_pop_ind_df = pd.merge(fn_pop_df, local_ind_df, on=["code", "year", "prefecture"],validate='one_to_one')

## Make sure only the prefectures have no counterpart
#fn_pop_df.loc[~fn_pop_df['code'].isin(fn_ind_df['code']) & ~(fn_pop_df['city_x']=='prefecture')]

fn_pop_ind_pref_df = pd.merge(fn_pop_pref_df, prefmean_ind_df, on=["year","prefecture"],validate='one_to_one')
#fn_pop_ind_pref_df = pd.merge(fn_pop_pref_df, pref_ind_df, on=["year","prefecture"],validate='one_to_one')

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

#######################
### Removing Tokyo ####
#######################

tokyo23codes = [str(13101 + i) for i in range(23)]
local_df_no23 = local_df.drop(local_df.loc[local_df['code'].isin(tokyo23codes)].index)

##############################
#### Output prefecture table  ###########
##############################

year = 2021
pref_df.loc[(pref_df['year'] == year),['prefecture', 'netgainminusdeductions']].sort_values(['netgainminusdeductions'])
pref_df.loc[(pref_df['year'] == year),['prefecture', 'profit-per-person']].sort_values(['profit-per-person'])
pref_df.loc[(pref_df['year'] == year),['prefecture', 'donations']].sort_values(['donations'])
pref_df.loc[(pref_df['year'] == year),['prefecture', 'total-pop']].sort_values(['total-pop'])

##############################
#### Output municipal table  #
##############################

year = 2021
local_df.loc[(local_df['year'] == year),['city', 'netgainminusdeductions']].sort_values(['netgainminusdeductions'])
local_df.loc[(local_df['year'] == year),['prefecture','city', 'profit-per-person', 'total-pop']].sort_values(['profit-per-person'])
local_df.loc[(local_df['year'] == year),['city', 'donations']].sort_values(['donations'])
local_df.loc[(local_df['year'] == year),['city', 'total-pop']].sort_values(['total-pop'])

test_df = local_df.loc[(local_df['year'] == year),['city', 'netgainminusdeductions']]
test_df['rank'] = test_df['netgainminusdeductions'].rank(ascending=False)
test_df.loc[test_df['city']=='厚岸町']
##############################
#### PROFIT AND POPULATION  ##
##############################

fig, ax = cb.handlers()
title = r"Population vs Profit"
#subtitle = r"(1m people)"
#ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
year = [2020]
ax.scatter(local_df.loc[local_df['year'].isin(year)]['total-pop'],local_df.loc[local_df['year'].isin(year)]['netgainminusdeductions'])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'profit_vs_pop.'+suffix, transparent=True)
plt.close('all')

fig = px.scatter(local_df.loc[local_df['year'].isin(year)], x='total-pop',y='netgainminusdeductions', color='prefecture', hover_data=['code','prefecture', 'city_x'])
fig.write_html(PLOT_FOLDER+'/profit_vs_pop.html')
plt.close('all')

##############################
#### PROFIT AND POP CHANGE  ##
##############################

fig, ax = cb.handlers()
title = r"Population Change vs Profit"
#subtitle = r"(1m people)"
#ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
year = [2020]
ax.scatter(local_df.loc[local_df['year'].isin(year)]['in-minus-out-rate'],local_df.loc[local_df['year'].isin(year)]['netgainminusdeductions'])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'profit_vs_popchange.'+suffix, transparent=True)
plt.close('all')

fig = px.scatter(local_df.loc[local_df['year'].isin(year)], x='in-minus-out-rate',y='netgainminusdeductions', color='prefecture', hover_data=['code','prefecture', 'city_x'])
fig.write_html(PLOT_FOLDER+'/profit_vs_popchange.html')
plt.close('all')

##############################
#### DONATIONS AND POP CHANGE  ##
##############################

fig, ax = cb.handlers()
title = r"Population Change vs Donations"
#subtitle = r"(1m people)"
#ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
year = [2020]
ax.scatter(local_df.loc[local_df['year'].isin(year)]['in-minus-out-rate'],local_df.loc[local_df['year'].isin(year)]['donations'])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'profit_vs_popchange.'+suffix, transparent=True)
plt.close('all')

fig = px.scatter(local_df.loc[local_df['year'].isin(year)], x='in-minus-out-rate',y='donations', color='prefecture', hover_data=['code','prefecture', 'city_x'])
fig.write_html(PLOT_FOLDER+'/profit_vs_popchange.html')
plt.close('all')

##############################
#### DONATIONS AND STRENGTH ##
##############################

fig, ax = cb.handlers()
title = r"Donations vs Strength"
#subtitle = r"(1m people)"
#ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
year = [2020]
ax.scatter(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'],local_df_no23.loc[local_df_no23['year'].isin(year)]['donations'])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'donations_vs_strength.'+suffix, transparent=True)
plt.close('all')

fig = px.scatter(local_df_no23.loc[local_df_no23['year'].isin(year)], x='economic-strength-index',y='donations', color='prefecture', hover_data=['code','prefecture', 'city_x'])
fig.write_html(PLOT_FOLDER+'/donations_vs_strength.html')
plt.close('all')

##############################
#### PROFIT AND STRENGTH  ##
##############################

fig, ax = cb.handlers()
title = r"Profit vs Strength"
#subtitle = r"(1m people)"
#ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
year = [2020]
ax.scatter(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'],local_df_no23.loc[local_df_no23['year'].isin(year)]['netgainminusdeductions'])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'profit_vs_strength.'+suffix, transparent=True)
plt.close('all')

fig = px.scatter(local_df_no23.loc[local_df_no23['year'].isin(year)], x='economic-strength-index',y='netgainminusdeductions', color='prefecture', hover_data=['code','prefecture', 'city_x'])
fig.write_html(PLOT_FOLDER+'/profit_vs_strength.html')
plt.close('all')

##############################
#### DONATIONS AND STRENGTH ーー PREF ##
##############################

fig, ax = cb.handlers()
title = r"Donations vs Strength"
#subtitle = r"(1m people)"
#ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
year = [2017]
ax.scatter(fn_pop_ind_pref_df.loc[fn_pop_ind_pref_df['year'].isin(year)]['economic-strength-index'],fn_pop_ind_pref_df.loc[fn_pop_ind_pref_df['year'].isin(year)]['donations'])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'donations_vs_strength_pref.'+suffix, transparent=True)
plt.close('all')

fig = px.scatter(fn_pop_ind_pref_df.loc[fn_pop_ind_pref_df['year'].isin(year)], x='economic-strength-index',y='donations', color='prefecture', hover_data=['prefecture'])
fig.write_html(PLOT_FOLDER+'/donations_vs_strength_pref.html')
plt.close('all')

##############################
#### PROFIT AND STRENGTH -- PREF   ##
##############################

fig, ax = cb.handlers()
title = r"Profit vs Strength"
#subtitle = r"(1m people)"
#ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
year = [2020]
ax.scatter(fn_pop_ind_pref_df.loc[fn_pop_ind_pref_df['year'].isin(year)]['economic-strength-index'],fn_pop_ind_pref_df.loc[fn_pop_ind_pref_df['year'].isin(year)]['netgainminusdeductions'])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'profit_vs_strength.'+suffix, transparent=True)
plt.close('all')

fig = px.scatter(fn_pop_ind_pref_df.loc[fn_pop_ind_pref_df['year'].isin(year)], x='economic-strength-index',y='netgainminusdeductions', color='prefecture', hover_data=['prefecture'])
fig.write_html(PLOT_FOLDER+'/profit_vs_strength_pref.html')
plt.close('all')

##############################
#### DONATIONS AND STRENGTH HISTOGRAM ##
##############################

fig, ax = cb.handlers()
title = r"Donations vs Strength"
#subtitle = r"(1m people)"
#ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
year = [2018]
ax.hist(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], bins=20, density=True, color=samcolors.nice_colors(3))
ax.hist(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], bins=20, weights = local_df_no23.loc[local_df_no23['year'].isin(year)]['donations'], density=True, color=samcolors.nice_colors(0),alpha=.5)
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'donations_vs_strength_histogram.'+suffix, transparent=True)
plt.close('all')

##############################
#### DONATIONS AND STRENGTH PLOT ##
##############################

fig, ax = cb.handlers()
title = r"Donations vs Strength"
#subtitle = r"(1m people)"
#ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
year = [2020]#2016,2017,2018,2019,
from scipy.stats import binned_statistic
mean, _, _ = binned_statistic(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], local_df_no23.loc
[local_df_no23['year'].isin(year)]['donations'], statistic='mean', bins=20)
median, _, _ = binned_statistic(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], local_df_no23.loc
[local_df_no23['year'].isin(year)]['donations'], statistic='median', bins=20)
binmean, _, _ = binned_statistic(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'],local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], statistic='median', bins=20)
ax.scatter(binmean,mean, color='red')
ax.plot(binmean,median, color='blue')
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'donations_vs_strength_plot.'+suffix, transparent=True)
plt.close('all')

##############################
#### PROFIT AND STRENGTH PLOT ##
##############################

fig, ax = cb.handlers()
title = r"Profit vs Strength"
#subtitle = r"(1m people)"
#ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
year = [2020]#2016,2017,2018,2019,
from scipy.stats import binned_statistic
mean, _, _ = binned_statistic(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], local_df_no23.loc
[local_df_no23['year'].isin(year)]['netgainminusdeductions'], statistic='mean', bins=20)
median, _, _ = binned_statistic(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], local_df_no23.loc
[local_df_no23['year'].isin(year)]['netgainminusdeductions'], statistic='median', bins=20)
binmean, _, _ = binned_statistic(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'],local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], statistic='mean', bins=20)
ax.scatter(binmean,mean, color='red')
ax.scatter(binmean,median, color='blue')
ax.set_xlim([0.1,1.2])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'profit_vs_strength_plot.'+suffix, transparent=True)
plt.close('all')

##############################
#### PROFIT PER CAPITA AND STRENGTH PLOT ##
##############################

fig, ax = cb.handlers()
title = r"Profit Per Capita vs Strength"
#subtitle = r"(1m people)"
#ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
year = [2020]
from scipy.stats import binned_statistic
mean, _, _ = binned_statistic(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], local_df_no23.loc
[local_df_no23['year'].isin(year)]['profit-per-person'], statistic='mean', bins=20)
median, _, _ = binned_statistic(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], local_df_no23.loc
[local_df_no23['year'].isin(year)]['profit-per-person'], statistic='median', bins=20)
std, _, _ = binned_statistic(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], local_df_no23.loc
[local_df_no23['year'].isin(year)]['profit-per-person'], statistic='std', bins=20)
binmean, _, _ = binned_statistic(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'],local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], statistic='mean', bins=20)
ax.errorbar(binmean,mean, std,color='red')
ax.errorbar(binmean,median, std, color='blue')
ax.set_xlim([0.1,1.4])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'profit-per-person_vs_strength_plot.'+suffix, transparent=True)
plt.close('all')

###################################
#### STRENGTH OF TOP N DONATIONS ##
###################################

fn_rough_pop_df = pd.merge(furusato_rough_df, local_pop_df, on=["code", "year", "prefecture"],validate='one_to_one')
furusato_rough_df.loc[~furusato_rough_df['code'].isin(fn_rough_pop_df['code']) & ~(furusato_rough_df['city']=='prefecture')]

fn_rough_ind_df = pd.merge(fn_rough_pop_df, local_ind_df, on=["code", "year", "prefecture"],validate='one_to_one')
fn_rough_ind_df_no23 =  fn_rough_ind_df.drop(fn_rough_ind_df.loc[fn_rough_ind_df['code'].isin(tokyo23codes)].index)

topN = 20

langs = ['en','jp']

colorstring = (r"\definecolor{blue}{rgb}{"+str(samcolors.nice_colors(3))[1:-1]+"}"+
r"\definecolor{yellow}{rgb}{"+str(samcolors.nice_colors(1))[1:-1]+"}" +
r"\definecolor{orange}{rgb}{"+str(samcolors.nice_colors(0.5))[1:-1]+"}" +
r"\definecolor{red}{rgb}{"+str(samcolors.nice_colors(0))[1:-1]+"}" 
)

titles = {'en':r"\textbf{The top winners are already relatively rich}", 'jp':r"\textbf{儲かっている自治体は比較的にもう裕福}"}
subtitles = {'en':colorstring+r"Economic strength index of \textbf{\textcolor{blue}{top " + str(topN) + " municipalities}}", 'jp':
colorstring+r"寄附金額が多い\textbf{\textcolor{blue}{２０自治体}}の財政力指標"}

for lang in langs:
    fig, ax = cb.handlers()
    cb.set_titleSubtitle(ax, titles[lang], subtitles[lang])
    cb.set_yTickLabels(ax)
    # cb.set_source(ax, "Data: Ministry of Internal Affairs",loc='outside')
    # cb.set_byline(ax, "by Sam Passaglia")

    names_list = []
    topN_economic_list=[]
    all_economic_list = []
    for i, year in enumerate(fn_rough_ind_df.year.unique()):
        print(year)
        inds = np.argsort(fn_rough_ind_df.loc[fn_rough_ind_df['year']==year,'donations'].values)[::-1]
        names_list.append(fn_rough_ind_df.loc[fn_rough_ind_df['year']==year,'city'].iloc[inds[0:topN]].values)
        topN_economic_list.append(fn_rough_ind_df.loc[fn_rough_ind_df['year']==year,'economic-strength-index'].iloc[inds[0:topN]].values)
        all_economic_list.append(fn_rough_ind_df_no23.loc[fn_rough_ind_df_no23['year']==year,'economic-strength-index'])
        
    ax.plot(fn_rough_ind_df.year.unique(), [np.mean(topN_economic_list[i]) for i in range(len(topN_economic_list))], color=samcolors.nice_colors(3))
    ax.plot(fn_rough_ind_df.year.unique(),  [np.quantile(all_economic_list[i],.5) for i in range(len(all_economic_list))], color=samcolors.nice_colors(1))
    ax.plot(fn_rough_ind_df.year.unique(),  [np.quantile(all_economic_list[i],.25) for i in range(len(all_economic_list))], color=samcolors.nice_colors(0.5))
    #ax.plot(fn_rough_ind_df.year.unique(),  [np.quantile(all_economic_list[i],.01) for i in range(len(all_economic_list))], color=samcolors.nice_colors(0))
    ax.plot(fn_rough_ind_df.year.unique(),  [np.mean(np.sort(all_economic_list[i])[0:20]) for i in range(len(all_economic_list))], color=samcolors.nice_colors(0))

    ax.set_xlim(min(furusato_rough_df.year)-.8,max(furusato_rough_df.year)+.2)
    ax.set_xticks([2009,2011,2013,2015,2017,2019,2021])

    if lang == 'en':
        ax.text(2008, .5, colorstring+r'\textbf{\textcolor{yellow}{All-Japan median}}', va='bottom', ha='left', fontsize=14)
        ax.text(2008, .29, colorstring+r'\textbf{\textcolor{orange}{25th percentile: One quarter lie below this line}}', va='bottom', ha='left', fontsize=14)
        ax.text(2008, .1, colorstring+r'\textbf{\textcolor{red}{Economically weakest 20 municipalities}}', va='bottom', ha='left', fontsize=14)
    elif lang == 'jp':
        ax.text(2008, .52, colorstring+r'\textbf{\textcolor{yellow}{全国中央値}}', va='bottom', ha='left', fontsize=14)
        ax.text(2008, .31, colorstring+r'\textbf{\textcolor{orange}{２５パーセンタイル：この線の下には自治体の４分の１}}', va='bottom', ha='left', fontsize=14)
        ax.text(2008, .12, colorstring+r'\textbf{\textcolor{red}{財政力最弱２０自治体}}', va='bottom', ha='left', fontsize=14)

    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.1f}'.format))

    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+'topN-strength_'+lang+'.'+suffix, transparent=True, bbox_inches='tight')
    shutil.copy(PLOT_FOLDER+'topN-strength_'+lang+'.pdf',EXTRA_PLOT_FOLDER)

    plt.close('all')

# ##############################
# #### DONATIONS AND BURDEN HISTOGRAM ##
# ##############################

# fig, ax = cb.handlers()
# title = r"Donations vs Burden"
# #subtitle = r"(1m people)"
# #ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
# fig.suptitle(title, x=0,y=1.1, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
# year = [2020]
# ax.hist(local_df_no23.loc[local_df_no23['year'].isin(year)]['future-burden-rate'], bins=20, density=True, color=samcolors.nice_colors(3))
# ax.hist(local_df_no23.loc[local_df_no23['year'].isin(year)]['future-burden-rate'], bins=20, weights = local_df_no23.loc[local_df_no23['year'].isin(year)]['donations'], density=True, color=samcolors.nice_colors(0),alpha=.5)
# for suffix in output_filetypes:
#     fig.savefig(PLOT_FOLDER+'donations_vs_burden_histogram.'+suffix, transparent=True)
# plt.close('all')

# ##############################
# #### DONATIONS AND laspeyres ##
# ##############################

# fig, ax = cb.handlers()
# title = r"Donations vs laspeyres"
# #subtitle = r"(1m people)"
# #ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
# fig.suptitle(title, x=0,y=1.1, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
# year = [2020]
# ax.hist(local_df_no23.loc[local_df_no23['year'].isin(year)]['laspeyres'], bins=20, density=True, color=samcolors.nice_colors(3))
# ax.hist(local_df_no23.loc[local_df_no23['year'].isin(year)]['laspeyres'], bins=20, weights = local_df_no23.loc[local_df_no23['year'].isin(year)]['donations'], density=True, color=samcolors.nice_colors(0),alpha=.5)
# for suffix in output_filetypes:
#     fig.savefig(PLOT_FOLDER+'donations_vs_laspeyres_histogram.'+suffix, transparent=True)
# plt.close('all')

# ##############################
# #### DONATIONS AND expense ##
# ##############################

# fig, ax = cb.handlers()
# title = r"Donations vs expense"
# #subtitle = r"(1m people)"
# #ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
# fig.suptitle(title, x=0,y=1.1, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
# year = [2020]
# ax.hist(local_df_no23.loc[local_df_no23['year'].isin(year)]['regular-expense-rate'], bins=20, density=True, color=samcolors.nice_colors(3))
# ax.hist(local_df_no23.loc[local_df_no23['year'].isin(year)]['regular-expense-rate'], bins=20, weights = local_df_no23.loc[local_df_no23['year'].isin(year)]['donations'], density=True, color=samcolors.nice_colors(0),alpha=.5)
# for suffix in output_filetypes:
#     fig.savefig(PLOT_FOLDER+'donations_vs_expense_histogram.'+suffix, transparent=True)
# plt.close('all')


# ##############################
# #### PROFIT AND STRENGTH HISTOGRAM ##
# ##############################

# fig, ax = cb.handlers()
# title = r"Profit vs Strength"
# #subtitle = r"(1m people)"
# #ax.set_title(subtitle, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
# fig.suptitle(title, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
# year = [2020]
# #ax.hist(fn_ind_df.loc[fn_ind_df['year'].isin(year)]['economic-strength-index'])
# ax.hist(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], bins=20, color=samcolors.nice_colors(3))
# ax.hist(local_df_no23.loc[local_df_no23['year'].isin(year)]['economic-strength-index'], bins=20,weights = local_df_no23.loc[fn_ind_df['year'].isin(year)]['netgainminusdeductions']/(3*oku), color=samcolors.nice_colors(0),alpha=.5)
# for suffix in output_filetypes:
#     fig.savefig(PLOT_FOLDER+'profit_vs_strength_histogram.'+suffix, transparent=True)
# plt.close('all')

