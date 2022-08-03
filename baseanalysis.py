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
from japandata.furusatonouzei.data import furusato_arr, furusato_df, furusato_pref_df, furusato_rough_df
import shutil
from matplotlib.ticker import FuncFormatter,StrMethodFormatter


PLOT_FOLDER = os.path.join(os.path.dirname(__file__),'basicgraphics/')
EXTRA_PLOT_FOLDER = './manuscript/draft/figures/'

output_filetypes = ['pdf','png']

os.makedirs(PLOT_FOLDER, exist_ok=True)
for filetype in output_filetypes:
    os.makedirs(PLOT_FOLDER+'/'+filetype, exist_ok=True)

oku = 10**8

###################################
##### Summing over years ##########
###################################

furusato_sum_df = furusato_df.groupby(['code','prefecturecity','prefecture', 'city']).sum().reset_index().drop('year', axis=1)
furusato_pref_sum_df = furusato_pref_df.groupby(['prefecture']).sum().reset_index().drop('year', axis=1)

############
############

fig, ax = init_plotting(style='nyt')
title_string = r"Furusato nouzei: A program on the rise"
subtitle_string = r"Received, Reported, Deducted, Products (¥100m)"
ax.set_title(subtitle_string, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
fig.suptitle(title_string, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
ax.set_xlim(min(furusato_arr.year)-.5,max(furusato_arr.year)+.5)
ax.plot(furusato_arr.year, furusato_arr['donations'].sum(dim='prefecturecity')/oku, label= 'Municipalities Received')
ax.plot(furusato_arr.year, furusato_arr['reported-donations'].sum(dim='prefecturecity')/oku, label= 'People Reported Donating')
ax.plot(furusato_arr.year, furusato_arr['deductions'].sum(dim='prefecturecity')/oku, label= 'Deductions Received')
ax.plot(furusato_arr.year, (furusato_arr['product-cost']+furusato_arr['shipping-cost']).sum(dim='prefecturecity')/oku, label= 'Value of Products Received')
ax.set_ylim([0,7000])
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'donations-report-loss.'+suffix, transparent=True)
plt.close('all')

############
############

title_strings = {'en':r"\textbf{Furusato n\={o}zei is exploding in popularity}", 
'jp':r"\textbf{ふるさと納税の急速な拡大}"}
# 'jp':r"{ふるさと納税の急速な拡大}"}
scalings = {'en': 10**9, 'jp': 10**8}
subtitle_strings ={'en':r"Total donations (¥bn)", 
'jp':r'合計寄附金額（億円）'}
xlabels = {'en':'', 
'jp':r''} 

langs = ['en','jp']
predatayears = list(range(2008,2016))
predatadonations = list(np.array([81.4,77.0,102.2,121.6,104.1,145.6,388.5,1652.9])*10**8) # yen
roughsum = furusato_rough_df.groupby('year').sum()

for lang in langs: 
    fig, ax = init_plotting(style='nyt')
    ax.set_title(subtitle_strings[lang], x=0., y=1.01, fontsize=14,ha='left',va='bottom')
    fig.suptitle(title_strings[lang], x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
    ax.set_xlim(min(predatayears)-.5,max(furusato_arr.year)+.5)
    ax.set_xticks([2009,2011,2013,2015,2017,2019,2021])
    ax.plot(predatayears+list(furusato_arr.year), np.array(predatadonations+list(furusato_arr['donations'].sum(dim='prefecturecity')))/scalings[lang], color=samcolors.nice_colors(3))
    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.set_ylim([0,905 * scalings['en']/scalings[lang]])
    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+suffix+'/'+'total_donations_' + lang + '.'+suffix, transparent=True,bbox_inches='tight' )
    plt.close('all')

1-((np.array(predatadonations+list(furusato_arr['donations'].sum(dim='prefecturecity')))/scalings[lang])[1:]/
(np.array(predatadonations+list(furusato_arr['donations'].sum(dim='prefecturecity')))/scalings[lang])[:-1])

furusato_arr['donations'].sum(dim='prefecturecity')[-1].values /(100*10**9)

shutil.copy(PLOT_FOLDER+'pdf/total_donations_en.pdf',EXTRA_PLOT_FOLDER)
shutil.copy(PLOT_FOLDER+'pdf/total_donations_jp.pdf',EXTRA_PLOT_FOLDER)

############
############

title_strings = {'en':r"\textbf{Half of the money is lost to expenses}", 
'jp':r"\textbf{寄附金額の半分は費用に}"}
colorstring = (r"\definecolor{blue}{rgb}{"+str(samcolors.nice_colors(3))[1:-1]+"}"+
r"\definecolor{green}{rgb}{"+str(samcolors.nice_colors(4))[1:-1]+"}" +
r"\definecolor{red}{rgb}{"+str(samcolors.nice_colors(0))[1:-1]+"}" 
)
subtitle_strings ={'en':colorstring+r"Fraction of donation money going to the \textbf{\textcolor{blue}{cost of gifts}}," + "\n"+ colorstring+r"\textbf{\textcolor{green}{shipping}}, and \textbf{\textcolor{red}{administrative costs, marketing, and fees}}", 
'jp':colorstring+r"寄附金額のうち、\textbf{\textcolor{blue}{返礼品の調達}}、\textbf{\textcolor{green}{送付}}、"+ "\n" + colorstring+r"\textbf{\textcolor{red}{事務、 広報、 決済等}}、にかかる費用の割合"}
		
xlabels = {'en':'', 
'jp':r''} 

langs = ['en','jp']

for lang in langs: 
    fig, ax = init_plotting(style='nyt')
    ax.set_title(subtitle_strings[lang], x=0., y=1.01, fontsize=14,ha='left',va='bottom',wrap=True)
    fig.suptitle(title_strings[lang], x=0,y=1.25, fontsize=18,ha='left',va='bottom', transform=ax.transAxes, wrap=True)
    ax.set_xlim(min(furusato_arr.year)-.1,max(furusato_arr.year))
    ax.plot(list(furusato_arr.year), list(100*furusato_arr['product-cost'].sum(dim='prefecturecity')/furusato_arr['donations'].sum(dim='prefecturecity')), color=samcolors.nice_colors(3))
    ax.fill_between(list(furusato_arr.year),0, list(100*furusato_arr['product-cost'].sum(dim='prefecturecity')/furusato_arr['donations'].sum(dim='prefecturecity')), color=samcolors.nice_colors(3),alpha=.5)

    ax.plot(list(furusato_arr.year), list(100*(furusato_arr['product-cost'].sum(dim='prefecturecity')+furusato_arr['shipping-cost'].sum(dim='prefecturecity'))/furusato_arr['donations'].sum(dim='prefecturecity')), color=samcolors.nice_colors(4))
    ax.fill_between(list(furusato_arr.year), list(100*furusato_arr['product-cost'].sum(dim='prefecturecity')/furusato_arr['donations'].sum(dim='prefecturecity')),list(100*(furusato_arr['product-cost'].sum(dim='prefecturecity')+furusato_arr['shipping-cost'].sum(dim='prefecturecity'))/furusato_arr['donations'].sum(dim='prefecturecity')), color=samcolors.nice_colors(4),alpha=.5)

    ax.plot(list(furusato_arr.year), list(100*furusato_arr['total-cost'].sum(dim='prefecturecity')/furusato_arr['donations'].sum(dim='prefecturecity')), color=samcolors.nice_colors(0))
    ax.fill_between(list(furusato_arr.year), list(100*(furusato_arr['product-cost'].sum(dim='prefecturecity')+furusato_arr['shipping-cost'].sum(dim='prefecturecity'))/furusato_arr['donations'].sum(dim='prefecturecity')),list(100*furusato_arr['total-cost'].sum(dim='prefecturecity')/furusato_arr['donations'].sum(dim='prefecturecity')), color=samcolors.nice_colors(0),alpha=.5)

    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.0f}\%'.format))
    ax.set_xticks(list(furusato_arr.year))
    ax.set_ylim([0,60])
    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+suffix+'/'+'cost_' + lang + '.'+suffix, transparent=True,bbox_inches='tight')
    plt.close('all')

shutil.copy(PLOT_FOLDER+'pdf/cost_en.pdf',EXTRA_PLOT_FOLDER)
shutil.copy(PLOT_FOLDER+'pdf/cost_jp.pdf',EXTRA_PLOT_FOLDER)

############
############

### This is not correct because there is a deduction on the shotokuzei that is not reported by the city govs
# fig, ax = init_plotting(style='nyt')
# title_string = r"Furusato nouzei: A net gain for participants"
# subtitle_string = r"Value of received goods and rebates," "\n" "minus money donated (¥100m)"
# ax.set_title(subtitle_string, x=0., y=1.05, fontsize=14,ha='left',va='bottom')
# fig.suptitle(title_string, x=0,y=1.32, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
# ax.set_ylim([0, +500])
# ax.set_xlim(min(furusato_arr.year)-.5,max(furusato_arr.year)+.5)
# ax.axhline(0, color='black')
# ax.set_axisbelow(True)
# cmap = lambda val: samcolors.ColorsMarcoBlue if val>0 else samcolors.ColorsMarcoRed
# gains_by_people = (furusato_arr['deductions']+furusato_arr['product-cost']+furusato_arr['shipping-cost']-furusato_arr['donations']).sum(dim='prefecturecity')/oku
# pos_inds = np.where(gains_by_people>=0)[0]
# pos_years = furusato_arr.year[pos_inds]
# pos_gains_by_people = gains_by_people[pos_inds]
# pos_bar = ax.bar(pos_years, pos_gains_by_people, color = [cmap(val) for val in pos_gains_by_people])
# ax.bar_label(pos_bar,fmt='%.0f',padding=2)
# neg_inds = np.where(gains_by_people<0)[0]
# neg_years = furusato_arr.year[neg_inds]
# neg_gains_by_people = gains_by_people[neg_inds]
# neg_bar = ax.bar(neg_years, neg_gains_by_people, color = [cmap(val) for val in neg_gains_by_people])
# ax.bar_label(neg_bar,fmt='%.0f',padding=3)
# ax.spines['bottom'].set_color('none')
# ax.tick_params(axis=u'x', which=u'both',length=0, pad=5)
# for suffix in output_filetypes:
#     fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'net-donations-by-people.'+suffix, transparent=True)
# plt.close('all')

############
############
### This is not correct because there is a deduction on the shotokuzei that is not reported by the city govs
# fig, ax = init_plotting(style='nyt')
# title_string = r"People befuddled by overcomplicated system"
# subtitle_string = r"Losses from not reporting or overspending (¥100m)"
# ax.set_title(subtitle_string, x=0., y=1.05, fontsize=14,ha='left',va='bottom')
# fig.suptitle(title_string, x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
# ax.set_ylim([-2200,0])
# ax.set_xlim(min(furusato_arr.year)-.5,max(furusato_arr.year)+.5)
# ax.set_axisbelow(True)
# loss_to_not_reporting = (furusato_arr['reported-donations']-furusato_arr['donations']).sum(dim='prefecturecity')/oku
# loss_to_overspending = (furusato_arr['deductions']+2000*furusato_arr['reported-people']-furusato_arr['donations']).sum(dim='prefecturecity')/oku
# smallerbar = ax.bar(furusato_arr.year, loss_to_not_reporting, color = samcolors.nice_colors(1),label='Not Reporting',zorder=20)
# bar = ax.bar(furusato_arr.year, loss_to_overspending, color = samcolors.nice_colors(0),label='Overspending',zorder=10)
# ax.bar_label(bar,fmt='%.0f',padding=3)    
# ax.axhline(0, color='black', zorder=30)
# ax.legend(loc='lower left')
# ax.spines['bottom'].set_color('none')
# ax.tick_params(axis=u'x', which=u'both',length=0, pad=15)
# for suffix in output_filetypes:
#     fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'mistakes.'+suffix, transparent=True)
# plt.close('all')

############
############

fig, ax = init_plotting(style='nyt')
title_string = r"Loss Percentile "
subtitle_string = r" (¥100m)"
ax.set_title(subtitle_string, x=0., y=1.05, fontsize=14,ha='left',va='bottom')
fig.suptitle(title_string, x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
colors = [samcolors.nice_colors(i/len(furusato_arr.year)) for i in range(len(furusato_arr.year))]
for i, year in enumerate(furusato_arr.year):
    vals = furusato_arr.loc[{'year':year}]['netgainminusdeductions'].values
#    vals = furusato_arr.loc[{'year':year}]['donations'].values-furusato_arr.loc[{'year':year}]['deductions'].values
    vals = vals[np.where(np.isnan(vals)==False)]
    inds = np.argsort(vals)
    ax.plot(np.arange(1, len(vals)+1)/len(vals)*100,np.abs(vals[inds]/oku), color=colors[i], label=year.values)
    print(year.values)
    print(len(vals))
    print(np.round(100*(1-len(vals[np.where(vals>0)])/len(vals))), '% of places lose money')
    print(len(vals[np.where(vals<0)]))
ax.legend()
ax.set_yscale('log')
ax.set_axisbelow(True)
ax.set_ylim([10**-5,10**2])
ax.axhline(0)
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'percentile.'+suffix, transparent=True)
plt.close('all')

############
############

fig, ax = init_plotting(style='nyt')
title_string = r'Loss/Gain Distribution'
subtitle_string = r" (¥100m)"
ax.set_title(subtitle_string, x=0., y=1.05, fontsize=14,ha='left',va='bottom')
fig.suptitle(title_string, x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
colors = [samcolors.nice_colors(i/len(furusato_arr.year)) for i in range(len(furusato_arr.year))]
for i, year in enumerate(furusato_arr.year):
    net = furusato_arr.loc[{'year':year}]['netgainminusdeductions'].values
    net = net[~np.isnan(net)]
    inds = np.argsort(net)
    ax.hist(net[inds]/oku,density=True, bins=50,color=colors[i], label=year)
ax.legend()
ax.set_yscale('log')
ax.set_axisbelow(True)
ax.axhline(0)
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'histogram.'+suffix, transparent=True)
plt.close('all')

############
############

fig, ax = init_plotting(style='nyt')
title_string = r"Cumulative Donations"
subtitle_string = r"Fraction"
ax.set_title(subtitle_string, x=0., y=1.05, fontsize=14,ha='left',va='bottom')
fig.suptitle(title_string, x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
colors = [samcolors.nice_colors(i/len(furusato_arr.year)) for i in range(len(furusato_arr.year))]
for i, year in enumerate(furusato_arr.year):
    donations = furusato_arr.loc[{'year':year}]['donations'].values
    donations = np.sort(donations[~np.isnan(donations)])
    ax.plot(np.arange(1, len(donations)+1)/len(donations)*100,donations.cumsum()/donations.sum(), color=colors[i], label=year.values)
ax.legend()
ax.set_axisbelow(True)
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'cumulative_furusato_arr.'+suffix, transparent=True)
plt.close('all')

############
############

fig, ax = init_plotting(style='nyt')
donation_fraction = .1
title_string = r"Most gains go to just a few places"
subtitle_string = r"Fraction of municipalities getting " + str(int(donation_fraction*100)) + "\% of donations"
ax.set_title(subtitle_string, x=0., y=1.05, fontsize=14,ha='left',va='bottom')
fig.suptitle(title_string, x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
ax.set_xlim(min(furusato_arr.year)-.5,max(furusato_arr.year)+.5)
ax.set_axisbelow(True)
fraction_list = []
for i, year in enumerate(furusato_arr.year):
    donations = furusato_arr.loc[{'year':year}]['donations'].values
    donations = np.sort(donations[~np.isnan(donations)])[::-1]
    total_donations = donations.sum()
    number_of_towns = np.where((donations.cumsum()-donation_fraction*total_donations) > 0)[0][0]+1
    fraction_list.append(number_of_towns/len(donations))
ax.bar(furusato_arr.year, fraction_list, color =samcolors.ColorsMarcoBlue)
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'donation_fraction.'+suffix, transparent=True)
plt.close('all')

############
############

fig, ax = init_plotting(style='nyt')
donation_fraction = .1
title_string = r"Most gains go to just a few places"
subtitle_string = r"Number of municipalities getting " + str(int(donation_fraction*100)) + "\% of donations"
ax.set_title(subtitle_string, x=0., y=1.05, fontsize=14,ha='left',va='bottom')
fig.suptitle(title_string, x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
ax.set_xlim(min(furusato_arr.year)-.5,max(furusato_arr.year)+.5)
ax.set_axisbelow(True)
number_list = []
for i, year in enumerate(furusato_arr.year):
    donations = furusato_arr.loc[{'year':year}]['donations'].values
    donations = np.sort(donations[~np.isnan(donations)])[::-1]
    total_donations = donations.sum()
    number_of_towns = np.where((donations.cumsum()-donation_fraction*total_donations) > 0)[0][0]+1
    number_list.append(number_of_towns)
ax.bar(furusato_arr.year, number_list, color =samcolors.ColorsMarcoBlue)
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'donation_number.'+suffix, transparent=True)
plt.close('all')

############
############

topN = 20
langs = ['en', 'jp']
title_strings = {'en':r"\textbf{Winners take a huge chunk of the money}",'jp':r"\textbf{トップ自治体は寄附金の莫大なシェアを占める}"}
subtitle_strings = {'en':r"Proportion of donations going to \textbf{top " + str(int(topN)) + "} local governments (of "  + str(np.sum(~np.isnan(furusato_arr.loc[{'year':2020}]['donations'].values))) + ')','jp':r"受入額が多い\textbf{"+str(int(topN))+r"}自治体が占める全寄附金の割合（合計" + str(np.sum(~np.isnan(furusato_arr.loc[{'year':2020}]['donations'].values))) + 'ヶ所）'}#ふるさと納税

for lang in langs:
    fig, ax = init_plotting(style='nyt')
    ax.set_title(subtitle_strings[lang], x=0., y=1.05, fontsize=13,ha='left',va='bottom')
    fig.suptitle(title_strings[lang], x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
    ax.set_ylim(0,51)
    ax.set_xlim(min(furusato_rough_df.year)-.75,max(furusato_rough_df.year)+.75)

    fraction_list = []
    for i, year in enumerate(furusato_rough_df.year.unique()):
        donations = furusato_rough_df.loc[furusato_rough_df['year']==year]['donations'].values
        donations = np.sort(donations[~np.isnan(donations)])[::-1]
        total_donations = donations.sum()
        #print(year, ' Herfindahl–Hirschman Index: ', np.sum((np.array(donations)/total_donations)**2))
        #print(year, ' Herfindahl–Hirschman Index Percentage Version: ', np.sum((np.array(donations)/total_donations*100)**2))    
        fraction_list.append(np.sum(donations[:topN]/total_donations))
    ax.bar(furusato_rough_df.year.unique(), np.array(fraction_list)*100, color =samcolors.nice_colors(3))
    ax.set_xticks([2009,2011,2013,2015,2017,2019,2021])
    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.0f}\%'.format))
    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'donation_topN_'+lang+'.'+suffix, transparent=True, bbox_inches='tight')
    plt.close('all')
    shutil.copy(PLOT_FOLDER+'pdf/donation_topN_'+lang+'.pdf',EXTRA_PLOT_FOLDER)

############
############

from japandata.readings.data import names_df

topN = 10
title_strings = {'en':r"\textbf{Municipalities which receive the most donations}",'jp':r"\textbf{ふるさと納税受入額の多い10自治体}"}

langs = ['en','jp']
for lang in langs:
    fig, ax = init_plotting(style='nyt',figsize=(6.75,4))
    fig.suptitle(title_strings[lang], y=.92, fontsize=14)
    code_list = []
    for i, year in enumerate(furusato_arr.year):
        inds = np.argsort(furusato_arr.loc[{'year':year}]['donations'].values)[::-1]
        code_list.append(furusato_arr.loc[{'year':year}]['code'][inds][0:topN].values)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)
    uniques,counts=np.unique(code_list,return_counts=True)
    colors = dict()
    totalcolorsneeded = len(np.where(counts>1)[0])
    j=0
    cmap=matplotlib.cm.get_cmap('tab20c')
    for i, unique in enumerate(uniques):
        if counts[i] == 1:
            colors[unique] = "w"
        else:
            colors[unique] = cmap(j/(totalcolorsneeded)*16/20)
            j+=1

    code_list = np.array(code_list).T
    cellColours = np.ones_like(code_list)
    for i in range(len(code_list)):
        for j in range(len(code_list[i])):
            cellColours[i,j] = colors[code_list[i,j]]
        
    if lang == 'en':
        colname = 'city-reading'
    elif lang =='jp':
        colname = 'city'

    table = ax.table(cellText=[[names_df.loc[names_df['code']==code_list[i][j], colname].values[0].title().replace('ou', 'o') for j in range(len(code_list[j]))] for i in range(len(code_list))],
            cellColours=cellColours.tolist(),
            colLabels=furusato_arr.year.values,
            rowLabels=[str(i+1) + r'.' for i in range(topN)],
            cellLoc='center',
            loc='center',
            fontsize=10.5)
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.1, 2)

    import six
    for k, cell in six.iteritems(table._cells):
        cell.set_edgecolor('white')
        if k[1] == -1:
            cell.set_text_props(ha='center')
        # if k[0] == 0 or k[1] < header_columns:
        #     cell.set_text_props(weight='bold', color='w')
        #     cell.set_facecolor(header_color)
        # else:
        #     cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    ax.axis('tight')
    ax.axis('off')
    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'topN_table_'+lang+'.'+suffix, transparent=True,bbox_inches='tight')
    plt.close('all')

    shutil.copy(PLOT_FOLDER+'pdf/topN_table_'+lang+'.pdf',EXTRA_PLOT_FOLDER)

############
############

# fig, ax = init_plotting(style='nyt')
# plt.rcParams['text.usetex'] = 'False'
# plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'
# topN = 10
# title_string = r"Bottom 10 Donations Ranking"
# fig.suptitle(title_string, y=.92, fontsize=14)
# city_list = []
# for i, year in enumerate(furusato_arr.year):
#     # inds = np.argsort(furusato_arr.loc[{'year':year}]['donations'].values)
#     # city_list.append(furusato_arr.loc[{'year':year}]['city'][inds][0:topN].values)
#     ignorelist = ['prefecture']
#     inds = np.argsort(furusato_df.loc[(furusato_df['year']==year.values) & ~furusato_df['city'].isin(ignorelist) ,'donations'].values)
#     city_list.append(furusato_df.loc[(furusato_df['year']==year.values) & ~furusato_df['city'].isin(ignorelist) ,'city'].iloc[inds].values[0:topN])
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# plt.box(on=None)
# uniques,counts=np.unique(city_list,return_counts=True)
# colors = dict()
# totalcolorsneeded = len(np.where(counts>1)[0])
# j=0
# cmap=matplotlib.cm.get_cmap('tab20c')
# for i, unique in enumerate(uniques):
#     if counts[i] == 1:
#         colors[unique] = "w"
#     else:
#         colors[unique] = cmap(j/(totalcolorsneeded)*16/20)
#         j+=1

# city_list = np.array(city_list).T
# cellColours = np.ones_like(city_list)
# for i in range(len(city_list)):
#     for j in range(len(city_list[i])):
#         cellColours[i,j] = colors[city_list[i,j]]
# table = ax.table(cellText=city_list,
#         cellColours=cellColours.tolist(),
#         colLabels=furusato_arr.year.values,
#         rowLabels=[str(i+1) + '.' for i in range(topN)],
#         cellLoc='center',
#         loc='center')
# table.auto_set_font_size(False)            
# table.set_fontsize(12)
# table.scale(1.5, 2)
# ax.axis('tight')
# ax.axis('off')
# for suffix in output_filetypes:
#     fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'bottomN_table.'+suffix, transparent=True)
# plt.close('all')

############
############

fig, ax = init_plotting(style='nyt')
plt.rcParams['text.usetex'] = 'False'
plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'
topN = 10
title_string = r"Net Gain Ranking"
fig.suptitle(title_string, y=.92, fontsize=14)
city_list = []
for i, year in enumerate(furusato_arr.year):
    inds = np.argsort(np.nan_to_num(furusato_arr.loc[{'year':year}]['netgainminusdeductions'].values))[::-1]
    city_list.append(furusato_arr.loc[{'year':year}]['city'][inds][0:topN].values)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.box(on=None)
uniques,counts=np.unique(city_list,return_counts=True)
colors = dict()
totalcolorsneeded = len(np.where(counts>1)[0])
j=0
cmap=matplotlib.cm.get_cmap('tab20c')
for i, unique in enumerate(uniques):
    if counts[i] == 1:
        colors[unique] = "w"
    else:
        colors[unique] = cmap(j/(totalcolorsneeded)*16/20)
        j+=1

city_list = np.array(city_list).T
cellColours = np.ones_like(city_list)
for i in range(len(city_list)):
    for j in range(len(city_list[i])):
        cellColours[i,j] = colors[city_list[i,j]]
table = ax.table(cellText=city_list,
        cellColours=cellColours.tolist(),
        colLabels=furusato_arr.year.values,
        rowLabels=[str(i+1) + '.' for i in range(topN)],
        cellLoc='center',
        loc='center')
table.auto_set_font_size(False)            
table.set_fontsize(12)
table.scale(1.5, 2)
ax.axis('tight')
ax.axis('off')
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'topN_net_table.'+suffix, transparent=True)
plt.close('all')

################
################
################

fig, ax = init_plotting(style='nyt')
ax.set_axisbelow(True)
year = [2020]
donations = furusato_pref_df.loc[furusato_pref_df['year'].isin(year),'donations']
profits = furusato_pref_df.loc[furusato_pref_df['year'].isin(year),'netgainminusdeductions']
ax.scatter(donations,profits)
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'profit_vs_donation.'+suffix, transparent=True)
plt.close('all')

import plotly.express as px
import plotly.graph_objects as go

year = [2016,2020]
#furusato_pref_df['fractionofdonations'] = 0
totalbyyear = furusato_pref_df.groupby('year').sum()['donations']
def fracfunc(row):
    #row['fractionofdonations']=row['donations']/totalbyyear[row['year']]
    return row['donations']/totalbyyear[row['year']]


furusato_pref_df['fractionofdonations'] = furusato_pref_df.apply(fracfunc,axis=1)

fig = px.line(furusato_pref_df.loc[furusato_pref_df['year'].isin(year)], y='fractionofdonations',x='netgainminusdeductions', color='prefecture',hover_data=['prefecture', 'year'],markers=True)
# fig = px.scatter(furusato_pref_sum_df, x='donations',y='netgainminusdeductions', color='prefecture',hover_data=['prefecture'])
fig.write_html(PLOT_FOLDER+'/profit_vs_donation.html')
plt.close('all')

############
############

furusato_df.loc[(furusato_df['city'] == '泉佐野市') & (furusato_df['year'] == 2021),'netgainminusdeductions']