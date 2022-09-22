import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from samplot.circusboy import CircusBoy
import samplot.colors as samcolors
import shutil
from matplotlib.ticker import FuncFormatter

from data import furusato_df, furusato_rough_df, local_df, pref_df, annual_df, rough_annual_df

from config import productionPlotFolder

PLOT_FOLDER = os.path.join(os.path.dirname(__file__),'basic/')

output_filetypes = ['pdf','png']

os.makedirs(PLOT_FOLDER, exist_ok=True)
#os.makedirs(productionPlotFolder, exist_ok=True)

for filetype in output_filetypes:
    os.makedirs(PLOT_FOLDER+'/'+filetype, exist_ok=True)

oku = 10**8

cb = CircusBoy(baseFont='Helvetica', cjkFont   ='Hiragino Maru Gothic Pro',titleFont='Helvetica',textFont='Helvetica', fontsize=12,figsize=(6,4))

####################################
## Total Donations Chart ###########
####################################

titles = {'en':r"\textbf{Furusato n\={o}zei is exploding in popularity}", 
'jp':r"\textbf{ふるさと納税の急速な拡大}"}
scalings = {'en': 10**9, 'jp': 10**8}
subtitles ={'en':r"Total donations by year", 
'jp':r'合計寄附金額'}
xlabels = {'en':'', 
'jp':r''} 

langs = ['en','jp']
predatayears = list(range(2008,2016))
predatadonations = list(np.array([81.4,77.0,102.2,121.6,104.1,145.6,388.5,1652.9])*10**8) # yen

for lang in langs: 
    fig, ax = cb.handlers()
    cb.set_titleSubtitle(ax, titles[lang], subtitles[lang])
    if lang =='en':
        cb.set_yLabel(ax, yLabel=r'bn', currency=r'\textyen')
    if lang =='jp':
        cb.set_yLabel(ax, yLabel=r'億円', currency=r'')   
    cb.set_yTickLabels(ax)
    cb.set_source(ax, "Data: Ministry of Internal Affairs",loc='outside')
    cb.set_byline(ax, "Sam Passaglia")
    ax.set_xlim(min(annual_df.year)-.5,max(annual_df.year)+.5)
    ax.set_xticks([2009,2011,2013,2015,2017,2019,2021])
    ax.plot(predatayears+list(annual_df.year), np.array(predatadonations+list(annual_df['donations']))/scalings[lang], color=samcolors.nice_colors(3))
    #ax.plot(rough_annual_df.year, rough_annual_df['donations']/scalings[lang], color=samcolors.nice_colors(3))
    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.set_ylim([0,831 * scalings['en']/scalings[lang]])
    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+suffix+'/'+'total_donations_' + lang + '.'+suffix, transparent=True,bbox_inches='tight')
    plt.close('all')

shutil.copy(PLOT_FOLDER+'pdf/total_donations_en.pdf',productionPlotFolder)
shutil.copy(PLOT_FOLDER+'pdf/total_donations_jp.pdf',productionPlotFolder)

####################################
## Expenses Chart ##################
####################################

titles = {'en':r"\textbf{Half of the money is lost to expenses}", 
'jp':r"\textbf{寄附金額の半分は費用に}"}
titles = {'en':r"\textbf{Half of the money is lost to expenses}", 
'jp':r"\textbf{寄附金額の半分は費用に}"}
colorstring = (r"\definecolor{blue}{rgb}{"+str(samcolors.nice_colors(3))[1:-1]+"}"+
r"\definecolor{green}{rgb}{"+str(samcolors.nice_colors(4))[1:-1]+"}" +
r"\definecolor{red}{rgb}{"+str(samcolors.nice_colors(0))[1:-1]+"}" 
)
subtitles ={'en':colorstring+r"Fraction of donation money going to the \textbf{\textcolor{blue}{cost of gifts}}," + "\n" + colorstring+r"\textbf{\textcolor{green}{shipping}}, and \textbf{\textcolor{red}{administrative costs, marketing, and fees}}", 
'jp':colorstring+r"寄附金額のうち、\textbf{\textcolor{blue}{返礼品の調達}}、\textbf{\textcolor{green}{送付}}、"+ "\n" + colorstring+r"\textbf{\textcolor{red}{事務、 広報、 決済等}}、にかかる費用の割合"}

xlabels = {'en':'', 
'jp':r''} 

langs = ['en','jp']

for lang in langs: 
    fig, ax = cb.handlers()
    cb.set_titleSubtitle(ax, titles[lang], subtitles[lang])
    cb.set_yTickLabels(ax)
    # cb.set_source(ax, "Data: Ministry of Internal Affairs",loc='outside')
    # cb.set_byline(ax, "Sam Passaglia")

    ax.set_xlim(min(annual_df.year)-.4,max(annual_df.year))
    ax.plot(list(annual_df.year), list(100*annual_df['product-cost']/annual_df['donations']), color=samcolors.nice_colors(3))
    ax.fill_between(list(annual_df.year),0, list(100*annual_df['product-cost']/annual_df['donations']), color=samcolors.nice_colors(3),alpha=.5)

    ax.plot(list(annual_df.year), list(100*(annual_df['product-cost']+annual_df['shipping-cost'])/annual_df['donations']), color=samcolors.nice_colors(4))
    ax.fill_between(list(annual_df.year), list(100*annual_df['product-cost']/annual_df['donations']),list(100*(annual_df['product-cost']+annual_df['shipping-cost'])/annual_df['donations']), color=samcolors.nice_colors(4),alpha=.5)

    ax.plot(list(annual_df.year), list(100*annual_df['total-cost']/annual_df['donations']), color=samcolors.nice_colors(0))
    ax.fill_between(list(annual_df.year), list(100*(annual_df['product-cost']+annual_df['shipping-cost'])/annual_df['donations']),list(100*annual_df['total-cost']/annual_df['donations']), color=samcolors.nice_colors(0),alpha=.5)

    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.0f}\%'.format))
    ax.set_xticks(list(annual_df.year))
    ax.set_ylim([0,57])
    #cb.set_yLabel(ax, yLabel=r'%', currency=r'')
    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+suffix+'/'+'cost_' + lang + '.'+suffix, transparent=True,bbox_inches='tight')
    plt.close('all')

shutil.copy(PLOT_FOLDER+'pdf/cost_en.pdf',productionPlotFolder)
shutil.copy(PLOT_FOLDER+'pdf/cost_jp.pdf',productionPlotFolder)

############################
## Loss Percentile Chart ###
############################

fig, ax = cb.handlers()
title_string = r"Loss Percentile "
subtitle_string = r" (¥100m)"
ax.set_title(subtitle_string, x=0., y=1.05, fontsize=14,ha='left',va='bottom')
fig.suptitle(title_string, x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
colors = [samcolors.nice_colors(i/len(annual_df.year)) for i in range(len(annual_df.year))]
for i, year in enumerate(annual_df.year):
    vals = furusato_df.loc[furusato_df['year']==year]['netgainminusdeductions'].values
    vals = vals[np.where(np.isnan(vals)==False)]
    inds = np.argsort(vals)
    ax.plot(np.arange(1, len(vals)+1)/len(vals)*100,np.abs(vals[inds]/oku), color=colors[i], label=year)
    print(year)
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

#####################################
### Loss/Gain Distribution Chart ####
#####################################

fig, ax = cb.handlers()
title_string = r'Loss/Gain Distribution'
subtitle_string = r" (¥100m)"
ax.set_title(subtitle_string, x=0., y=1.05, fontsize=14,ha='left',va='bottom')
fig.suptitle(title_string, x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
colors = [samcolors.nice_colors(i/len(annual_df.year)) for i in range(len(annual_df.year))]
for i, year in enumerate(annual_df.year):
    net = furusato_df.loc[furusato_df['year']==year, 'netgainminusdeductions'].values
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

######################################
### Cumulative Distribution Chart ####
######################################

fig, ax = cb.handlers()
title_string = r"Cumulative Donations"
subtitle_string = r"Fraction"
ax.set_title(subtitle_string, x=0., y=1.05, fontsize=14,ha='left',va='bottom')
fig.suptitle(title_string, x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
colors = [samcolors.nice_colors(i/len(annual_df.year)) for i in range(len(annual_df.year))]
for i, year in enumerate(annual_df.year):
    donations = furusato_df.loc[furusato_df['year']==year]['donations'].values
    donations = np.sort(donations[~np.isnan(donations)])
    ax.plot(np.arange(1, len(donations)+1)/len(donations)*100,donations.cumsum()/donations.sum(), color=colors[i], label=year)
ax.legend()
ax.set_axisbelow(True)
for suffix in output_filetypes:
    fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'cumulative_donations.'+suffix, transparent=True)
plt.close('all')


# ############
# ############

# fig, ax = cb.handlers()
# donation_fraction = .1
# title_string = r"Most gains go to just a few places"
# subtitle_string = r"Number of municipalities getting " + str(int(donation_fraction*100)) + "\% of donations"
# ax.set_title(subtitle_string, x=0., y=1.05, fontsize=14,ha='left',va='bottom')
# fig.suptitle(title_string, x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
# ax.set_xlim(min(furusato_df.year)-.5,max(furusato_df.year)+.5)
# ax.set_axisbelow(True)
# number_list = []
# for i, year in enumerate(annual_df.year):
#     donations = furusato_df.loc[furusato_df['year']==year]['donations'].values
#     donations = np.sort(donations[~np.isnan(donations)])[::-1]
#     total_donations = donations.sum()
#     number_of_towns = np.where((donations.cumsum()-donation_fraction*total_donations) > 0)[0][0]+1
#     number_list.append(number_of_towns)
# ax.bar(annual_df.year, number_list, color =samcolors.ColorsMarcoBlue)
# for suffix in output_filetypes:
#     fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'donation_number.'+suffix, transparent=True)
# plt.close('all')

############
############

topN = 20
langs = ['en', 'jp']
titles = {'en':r"\textbf{Winners take a huge chunk of the money}",'jp':r"\textbf{トップ自治体は寄附金の莫大なシェアを占める}"}
subtitles = {'en':r"Proportion of donations going to \textbf{top " + str(int(topN)) + "} local governments (of "  + str(np.sum(~np.isnan(furusato_df.loc[furusato_df['year']==2020]['donations'].values))) + ')','jp':r"受入額が多い\textbf{"+str(int(topN))+r"}自治体が占める全寄附金の割合（合計" + str(np.sum(~np.isnan(furusato_df.loc[furusato_df['year']==2020]['donations'].values))) + 'ヶ所）'}

for lang in langs:
    fig, ax = cb.handlers()
    cb.set_titleSubtitle(ax, titles[lang], subtitles[lang])
    cb.set_yTickLabels(ax)
    # cb.set_source(ax, "Data: Ministry of Internal Affairs",loc='outside')
    # cb.set_byline(ax, "Sam Passaglia")
    ax.set_ylim(0,51)
    ax.set_xlim(min(rough_annual_df.year)-1.7,max(rough_annual_df.year)+.75)

    fraction_list = []
    for i, year in enumerate(rough_annual_df.year):
        donations = furusato_rough_df.loc[furusato_rough_df['year']==year]['donations'].values
        donations = np.sort(donations[~np.isnan(donations)])[::-1]
        total_donations = donations.sum()
        #print(year, ' Herfindahl–Hirschman Index: ', np.sum((np.array(donations)/total_donations)**2))
        #print(year, ' Herfindahl–Hirschman Index Percentage Version: ', np.sum((np.array(donations)/total_donations*100)**2))    
        fraction_list.append(np.sum(donations[:topN]/total_donations))
    ax.bar(rough_annual_df.year, np.array(fraction_list)*100, color =samcolors.nice_colors(3))
    ax.set_xticks([2009,2011,2013,2015,2017,2019,2021])
    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.0f}\%'.format))
    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'donation_topN_'+lang+'.'+suffix, transparent=True, bbox_inches='tight')
    plt.close('all')
    shutil.copy(PLOT_FOLDER+'pdf/donation_topN_'+lang+'.pdf',productionPlotFolder)

############
############

from japandata.readings.data import names_df

topN = 10
titles = {'en':r"\textbf{Municipalities which receive the most donations}",'jp':r"\textbf{ふるさと納税受入額の多い10自治体}"}

langs = ['en','jp']
for lang in langs:
    cbtable = CircusBoy(baseFont='Helvetica', cjkFont='Hiragino Maru Gothic Pro',titleFont='Helvetica',textFont='Helvetica', fontsize=12,figsize=(6.75,4))
    fig, ax = cbtable.handlers()
    fig.suptitle(titles[lang], y=.92, fontsize=cbtable.titlesize)
    code_list = []
    for i, year in enumerate(annual_df.year):
        inds = np.argsort(furusato_df.loc[furusato_df['year']==year]['donations'].values)[::-1]
        code_list.append(furusato_df.loc[furusato_df['year']==year]['code'].iloc[inds][0:topN].values)
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
    table = ax.table(cellText=[[names_df.loc[names_df['code']==code_list[i][j], colname].values[0].title().replace('ou', r'\={o}') for j in range(len(code_list[j]))] for i in range(len(code_list))],
            cellColours=cellColours.tolist(),
            colLabels=annual_df.year,
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

    shutil.copy(PLOT_FOLDER+'pdf/topN_table_'+lang+'.pdf',productionPlotFolder)


############
############

fig, ax = cb.handlers()
plt.rcParams['text.usetex'] = 'False'
plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'
topN = 10
title_string = r"Net Gain Ranking"
fig.suptitle(title_string, y=.92, fontsize=14)
city_list = []
for i, year in enumerate(annual_df.year):
    inds = np.argsort(np.nan_to_num(furusato_df.loc[furusato_df['year']==year]['netgainminusdeductions'].values))[::-1]
    city_list.append(furusato_df.loc[furusato_df['year']==year]['city'].iloc[inds][0:topN].values)
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
        colLabels=annual_df.year,
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
