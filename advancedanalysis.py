import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from samplot.circusboy import CircusBoy
import samplot.colors as samcolors

import plotly.express as px
import plotly.graph_objects as go
import shutil
from matplotlib.ticker import FuncFormatter

from data import local_df, pref_df, annual_df, rough_df, rough_annual_df, local_df_no23, rough_df_no23

PLOT_FOLDER = os.path.join(os.getcwd(),'advancedgraphics/')
EXTRA_PLOT_FOLDER = './furusato-private/draft/figures/'

output_filetypes = ['pdf']

os.makedirs(PLOT_FOLDER, exist_ok=True)

oku = 10**8
hyakuman = 10**6

cb = CircusBoy(baseFont='Helvetica', cjkFont='Hiragino Maru Gothic Pro',titleFont='Helvetica',textFont='Helvetica', fontsize=12,figsize=(6,4))

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
cb.set_titleSubtitle(ax, title)
year = [2020]
ax.scatter(local_df.loc[local_df['year'].isin(year)]['total-pop'],local_df.loc[local_df['year'].isin(year)]['netgainminusdeductions'])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'profit_vs_pop.'+suffix, transparent=True)
plt.close('all')

fig = px.scatter(local_df.loc[local_df['year'].isin(year)], x='total-pop',y='netgainminusdeductions', color='prefecture', hover_data=['code','prefecture', 'city'])
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

fig = px.scatter(local_df.loc[local_df['year'].isin(year)], x='in-minus-out-rate',y='netgainminusdeductions', color='prefecture', hover_data=['code','prefecture', 'city'])
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

fig = px.scatter(local_df.loc[local_df['year'].isin(year)], x='in-minus-out-rate',y='donations', color='prefecture', hover_data=['code','prefecture', 'city'])
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

fig = px.scatter(local_df_no23.loc[local_df_no23['year'].isin(year)], x='economic-strength-index',y='donations', color='prefecture', hover_data=['code','prefecture', 'city'])
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

fig = px.scatter(local_df_no23.loc[local_df_no23['year'].isin(year)], x='economic-strength-index',y='netgainminusdeductions', color='prefecture', hover_data=['code','prefecture', 'city'])
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
ax.scatter(pref_df.loc[pref_df['year'].isin(year)]['economic-strength-index'],pref_df.loc[pref_df['year'].isin(year)]['donations'])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'donations_vs_strength_pref.'+suffix, transparent=True)
plt.close('all')

fig = px.scatter(pref_df.loc[pref_df['year'].isin(year)], x='economic-strength-index',y='donations', color='prefecture', hover_data=['prefecture'])
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
ax.scatter(pref_df.loc[pref_df['year'].isin(year)]['economic-strength-index'],pref_df.loc[pref_df['year'].isin(year)]['netgainminusdeductions'])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'profit_vs_strength.'+suffix, transparent=True)
plt.close('all')

fig = px.scatter(pref_df.loc[pref_df['year'].isin(year)], x='economic-strength-index',y='netgainminusdeductions', color='prefecture', hover_data=['prefecture'])
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
    for i, year in enumerate(rough_df.year.unique()):
        print(year)
        inds = np.argsort(rough_df.loc[rough_df['year']==year,'donations'].values)[::-1]
        names_list.append(rough_df.loc[rough_df['year']==year,'city'].iloc[inds[0:topN]].values)
        topN_economic_list.append(rough_df.loc[rough_df['year']==year,'economic-strength-index'].iloc[inds[0:topN]].values)
        all_economic_list.append(rough_df_no23.loc[rough_df_no23['year']==year,'economic-strength-index'])
        
    ax.plot(rough_df.year.unique(), [np.mean(topN_economic_list[i]) for i in range(len(topN_economic_list))], color=samcolors.nice_colors(3))
    ax.plot(rough_df.year.unique(),  [np.quantile(all_economic_list[i],.5) for i in range(len(all_economic_list))], color=samcolors.nice_colors(1))
    ax.plot(rough_df.year.unique(),  [np.quantile(all_economic_list[i],.25) for i in range(len(all_economic_list))], color=samcolors.nice_colors(0.5))
    #ax.plot(rough_df.year.unique(),  [np.quantile(all_economic_list[i],.01) for i in range(len(all_economic_list))], color=samcolors.nice_colors(0))
    ax.plot(rough_df.year.unique(),  [np.mean(np.sort(all_economic_list[i])[0:20]) for i in range(len(all_economic_list))], color=samcolors.nice_colors(0))

    ax.set_xlim(min(rough_df.year)-.8,max(rough_df.year)+.2)
    ax.set_xticks([2009,2011,2013,2015,2017,2019,2021])

    ## TODO THE MEDIANS AND PERCENTILES ARE ALL TOKYO-FREE
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


##############################
#### PER PERSON PROFIT-LOSS ##
##############################

year = [2016,2021]

titles = {'en':r"\textbf{Growing gap between winners and losers}", 
'jp':r"\textbf{都道府県の間に、利益格差が広がっている}"}
subtitles = {'en':r"Population and per-capita profit, " + str(year[0]) + r" to " + str(year[1]), 
'jp':r"各県の人口と一人当たり純利益の進化、" + str(year[0]) + r"から" + str(year[1])}
xlabels = {'en':r'Profit per capita  (\textyen/person)', 
'jp':r'一人当たり純利益 (円／人）'} 
#ylabels = {'en':'Population ', 
#'jp':r'人口'} 
langs = ['en','jp']

for lang in langs: 
    fig, ax = cb.handlers()
    if lang =='en':
        cb.set_yLabel(ax, yLabel=r' million people', currency=r'')
    if lang =='jp':
        cb.set_yLabel(ax, yLabel=r'百万人', currency=r'')   
    ax.set_yticks([0,5,10,15,20])
    ax.set_ylim([0,1.55*10])
    ax.set_xlim([-11000,21000])
    cb.set_titleSubtitle(ax, titles[lang], subtitles[lang])
    cb.set_yTickLabels(ax)

    ax.set_xlabel(xlabels[lang],color='black')
    #ax.set_ylabel(ylabels[lang],color='black')
    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    #ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))

    deltas_jp = {"東京都":[0,-.5], "神奈川県":[200,.5], "大阪府":[0,-.5], "愛知県":[-2200,.2], "埼玉県":[200,-.4], "千葉県":[-2000,0], "兵庫県":[0,-.5], "福岡県":[-100,.5],"北海道":[0,-.5], "静岡県":[0,.5], "茨城県":[900,0], "京都府":[-1100,-.3], "広島県":[0,+.5], "奈良県":[-1700,0],"佐賀県":[5900,-.4],"鹿児島県":[4900,0],"宮崎県":[5900,.5], "山形県":[4000,.5],"熊本県":[1500,.5],"新潟県":[1200,.55], "徳島県":[0,-.5], "高知県":[2000,-.5], "山梨県":[7000,-.5]}
    deltas_en = {"東京都":[0,-.5], "神奈川県":[200,.5], "大阪府":[0,-.5], "愛知県":[-2200,.1], "埼玉県":[200,-.4], "千葉県":[-2300,0], "兵庫県":[0,-.5], "福岡県":[-100,.5],"北海道":[0,.5], "静岡県":[0,.5], "茨城県":[1200,0], "京都府":[-200,-.4], "広島県":[0,+.5], "奈良県":[0,-.5],"佐賀県":[5900,-.4],"鹿児島県":[4900,.3],"宮崎県":[6400,.5], "山形県":[4000,.5],"熊本県":[2500,.4],"新潟県":[1800,.4], "徳島県":[0,-.5],  "高知県":[2000,-.4], "山梨県":[7000,-.5]}
    for prefecture in pref_df['prefecture'].unique(): 
        xs = pref_df.loc[(pref_df['prefecture']==prefecture) &(pref_df['year'].isin(year))]['profit-per-person'].values
        ys = pref_df.loc[(pref_df['prefecture']==prefecture) & (pref_df['year'].isin(year))]['total-pop'].values / 10**6
        #ax.plot(pref_df.loc[(pref_df['prefecture']==prefecture) &(pref_df['year'].isin(year))]['profitperperson'],pref_df.loc[(pref_df['prefecture']==prefecture) & (pref_df['year'].isin(year))]['total-pop'])
        if xs[1]<0:
            color = samcolors.nice_colors(3)
        else: 
            color = samcolors.nice_colors(0)
        ax.annotate("", xy=(xs[0], ys[0]), xytext=(xs[1], ys[1]),
                arrowprops=dict(arrowstyle="<-",shrinkA=0, shrinkB=0,color=color),fontsize=10)

        # try: va = vas[prefecture]
        # except KeyError: va = 'center'
        # try: ha = has[prefecture]
        # except KeyError: ha = 'center'
        if lang == 'jp':
            deltas = deltas_jp
        else:
            deltas = deltas_en
        try: delta = deltas[prefecture]
        except KeyError: delta = [0,0]

        if prefecture in deltas.keys():
            if lang == 'jp':
                label = prefecture.strip("県").strip("都").strip("府")
            else:
                label = pref_df.loc[pref_df['prefecture']==prefecture, 'prefecture-reading'].iloc[0]
                label = label.replace('ou','o').replace('oo','o').title()
            ax.text(np.mean(xs)+delta[0],np.mean(ys)+delta[1],label, fontsize=8, ha='center',va='center',color=cb.grey)
        
        if prefecture == '東京都':
            ax.text(xs[0],ys[0]+.15,year[0], fontsize=8, ha='center',va='bottom')
            ax.text(xs[1]+300,ys[1]-.2,year[1], fontsize=8, ha='center',va='top')

    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+'population_vs_profitperperson_'+lang+'.'+suffix, transparent=True,bbox_inches="tight")
    plt.close('all')

shutil.copy(PLOT_FOLDER+'population_vs_profitperperson_en.pdf',EXTRA_PLOT_FOLDER)
shutil.copy(PLOT_FOLDER+'population_vs_profitperperson_jp.pdf',EXTRA_PLOT_FOLDER)

import plotly.express as px
import plotly.graph_objects as go
fig = px.line(pref_df.loc[pref_df['year'].isin(year)], y='total-pop',x='profit-per-person', color='prefecture',hover_data=['prefecture', 'year'],markers=True)
# fig = px.scatter(furusato_pref_sum_df, x='donations',y='netgainminusdeductions', color='prefecture',hover_data=['prefecture'])
fig.write_html(PLOT_FOLDER+'/population_vs_profitperperson.html')
plt.close('all')

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
        pref_df.loc[pref_df['year'].isin(year)]['profit-per-person'] >= 0)
    neginds = np.where(
        pref_df.loc[pref_df['year'].isin(year)]['profit-per-person'] < 0)
    ax.scatter(pref_df.loc[pref_df['year'].isin(year)]['economic-strength-index'].iloc[posinds],
               pref_df.loc[pref_df['year'].isin(year)]['profit-per-person'].iloc[posinds], color=samcolors.nice_colors(0))
    ax.scatter(pref_df.loc[pref_df['year'].isin(year)]['economic-strength-index'].iloc[neginds],
               pref_df.loc[pref_df['year'].isin(year)]['profit-per-person'].iloc[neginds], color=samcolors.nice_colors(3))

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
    for prefecture in pref_df['prefecture'].unique():
        if lang == 'jp':
                label = prefecture.strip("県").strip("都").strip("府")
        else:
                label = pref_df.loc[pref_df['prefecture']
                                          == prefecture, 'prefecture-reading'].iloc[0]
                label = r'\phantom{h}'+label.replace('ou', 'o').replace('oo', 'o').title()+r'\phantom{g}'
        
        x = pref_df.loc[(pref_df['year'].isin(year)) & (
            pref_df['prefecture'] == prefecture)]['economic-strength-index']
        y = pref_df.loc[(pref_df['year'].isin(year)) & (
            pref_df['prefecture'] == prefecture)]['profit-per-person']
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

    #np.mean(pref_df.loc[(pref_df['year'].isin(year)) & (pref_df['prefecture']!='東京都')]['economic-strength-index'])
    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+'profitperperson_vs_strength_' +
                    lang+'.'+suffix, transparent=True, bbox_inches='tight')
    plt.close('all')
    shutil.copy(PLOT_FOLDER+'profitperperson_vs_strength_' +
                lang+'.pdf', EXTRA_PLOT_FOLDER)


fig = px.scatter(pref_df.loc[pref_df['year'].isin(
    year)], x='economic-strength-index', y='profit-per-person', color='prefecture', hover_data=['prefecture'])
fig.write_html(PLOT_FOLDER+'/profitperperson_vs_strength_pref.html')
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


# ##############################
# #### PROFIT VS STRENGTH PER PREF   ####
# ##############################

# ## TODO: MAKE THIS LESS SHIT AND DO DONATIONS TOO

# fig, ax = init_plotting(style='nyt')
# title_string = r""
# #subtitle_string = r"(1m people)"
# #ax.set_title(subtitle_string, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
# fig.suptitle(title_string, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
# years = [2019]
# #ax.scatter(local_df.loc[local_df['year'].isin(year)]['economic-strength-index'],local_df.loc[local_df['year'].isin(year)]['profitperperson'])
# #ax.scatter(local_df.loc[local_df['year'].isin(year)]['economic-strength-index'],local_df.loc[local_df['year'].isin(year)]['netgainminusdeductions'])
# ax.scatter(local_df.loc[local_df['year'].isin(years)]['economic-strength-index'],local_df.loc[local_df['year'].isin(years)]['donations-fraction'])
# #ax.scatter(local_df.loc[local_df['year'].isin(year)]['economic-strength-index'],local_df.loc[local_df['year'].isin(year)]['donations'])

# #ax.set_xlim([.2,1])
# #ax.set_ylim([-5000,20000])
# #ax.axhline(0,color='black', alpha=.5, ls='--')
# for suffix in output_filetypes:
#     fig.savefig(PLOT_FOLDER+PREFECTURE+'donationsfrac_vs_strength.'+suffix, transparent=True)
# plt.close('all')

# # fig = px.scatter(local_df.loc[local_df['year'].isin(year)], x='economic-strength-index',y='donations', color='prefecture', hover_data=['prefecture'])
# # fig.write_html(PLOT_FOLDER+PREFECTURE+'profitperperson_vs_strength_pref.html')
# # plt.close('all')