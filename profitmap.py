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


from japandata.population.data import local_pop_df, prefecture_pop_df
from japandata.furusatonouzei.data import furusato_arr, furusato_df, furusato_pref_df
from japandata.maps.data import load_map
from japandata.indices.data import local_ind_df, pref_ind_df
from japandata.readings.data import names_df, pref_names_df

PLOT_FOLDER = os.path.join(os.getcwd(),'profitmap/')
EXTRA_PLOT_FOLDER = './furusato-private/draft/figures/'

output_filetypes = ['pdf']

os.makedirs(PLOT_FOLDER, exist_ok=True)

oku = 10**8
hyakuman = 10**6

###################################
##### Merging pop data over years ##########
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

#######################################
###### Computing some useful things ###
#######################################
fn_pop_pref_df['profitperperson'] = fn_pop_pref_df.apply(lambda row: row['netgainminusdeductions']/row['total-pop'],axis=1)
totalbyyear = furusato_pref_df.groupby('year').sum()['donations']
fn_pop_pref_df['donationsfraction'] = fn_pop_pref_df.apply(lambda row: row['donations']/totalbyyear[row['year']],axis=1)
fn_pop_pref_df['donationsperperson'] = fn_pop_pref_df.apply(lambda row: row['donations']/row['total-pop'],axis=1)

###################################
##### Summing over years ##########
###################################

local_sum_df = fn_pop_df.groupby(['code','prefecturecity','prefecture', 'city_x']).sum().reset_index().drop('year', axis=1)
pref_sum_df = fn_pop_pref_df.groupby(['prefecture']).sum().reset_index().drop('year', axis=1)

################
### Aliases ####
################

local_df = fn_pop_df
pref_df = fn_pop_pref_df

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

##############################
#### FOLIUM MAP OF PROFIT: PREFECTURE ###
##############################

mapyear = 2021
map_df = load_map(mapyear,level='prefecture', quality='stylized')
df = pref_df.loc[pref_df['year']==mapyear]
# df = pref_sum_df

datacolumn= "netgainminusdeductions"
datacolumnalias = "Net Gain Minus Deductions (百万円)"
scalingfactor = hyakuman

map_df[datacolumn] = 0
for i in range(len(map_df)):
    try: 
        map_df.loc[i, datacolumn] =  (df.loc[df['prefecture'] == map_df.loc[i,'prefecture'],datacolumn].values[0]/scalingfactor)
    except IndexError: 
        print(map_df.loc[i,'code'])
        print(map_df.loc[i,'city'])

#largestdeviation = np.max(np.abs(map_df[datacolumn]))
largestdeviation = 20000

rgbacolors = (matplotlib.cm.get_cmap('coolwarm')(np.linspace(0,1,11)))
colormap = cm.LinearColormap( [matplotlib.colors.to_hex(color) for color in rgbacolors],
    vmin=-largestdeviation, vmax=largestdeviation
)
#colormap = lambda x: samcolors.ColorsMarcoRed if x > 0 else samcolors.ColorsMarcoBlue

m = folium.Map(**map_style)

tooltip = GeoJsonTooltip(
    fields=["prefecture", datacolumn],
    aliases=["Prefecture:", datacolumnalias + ':'],
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
        return colormap(feature["properties"][datacolumn]) 
    except ValueError: 
        print(feature["properties"]["name"])
        return 'black'
    except KeyError:
        print(feature["properties"]["name"])
        return 'black'

folium.GeoJson(map_df
,name = datacolumn
,style_function=lambda feature: 
{'fillColor':fillColor(feature)} | unhighlighted_style
,highlight_function=lambda feature: 
{'fillColor':fillColor(feature)} | highlighted_style
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

mapyear = 2021
map_df = load_map(mapyear,level='local_dc')
df = local_df.loc[local_df['year']==mapyear]
#df = local_sum_df

datacolumn= "netgainminusdeductions"
datacolumnalias = "Net Gain Minus Deductions (百万円)"
scalingfactor = hyakuman

map_df[datacolumn] = 0
for i in range(len(map_df)):
    try: 
        map_df.loc[i, datacolumn] =  (df.loc[df['code'] == map_df.loc[i,'code'],datacolumn].values[0]/scalingfactor)
    except IndexError: 
        print(map_df.loc[i,'code'])
        print(map_df.loc[i,'city'])

map_df.loc[map_df['city'] ==  '双葉町'].code
df.loc[df['city_x'] == '双葉町'].code
#largestdeviation = np.max(np.abs(map_df[datacolumn]))
largestdeviation = 2000

rgbacolors = (matplotlib.cm.get_cmap('coolwarm')(np.linspace(0,1,11)))
colormap = cm.LinearColormap( [matplotlib.colors.to_hex(color) for color in rgbacolors],
    vmin=-largestdeviation, vmax=largestdeviation
)

m = folium.Map(**map_style)

tooltip = GeoJsonTooltip(
    fields=["prefecture", "city", datacolumn],
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
        return colormap(feature["properties"][datacolumn]) 
    except ValueError: 
        print(feature["properties"]["name"])
        return 'black'
    except KeyError:
        print(feature["properties"]["name"])
        return 'black'

folium.GeoJson(map_df
,name = datacolumn
,style_function=lambda feature: 
{'fillColor':fillColor(feature)} | unhighlighted_style
,highlight_function=lambda feature: 
{'fillColor':fillColor(feature)} | highlighted_style
,zoom_on_click=True
,tooltip=tooltip
).add_to(m)

colormap.caption = datacolumnalias
colormap.add_to(m)

m.save(PLOT_FOLDER+"profit_map.html")


##################################
### Matplotlib version -- pref ###
##################################

mapyear = 2021
map_df = load_map(mapyear,level='prefecture', quality='stylized')
national_map_df = load_map(mapyear,level='japan', quality='stylized')
df = pref_df.loc[furusato_df['year']==mapyear]
#df = pref_sum_df 

datacolumn= "profitperperson"
scalingfactor = 1

map_df[datacolumn] = 0
for i in range(len(map_df)):
    try: 
        map_df.loc[i, datacolumn] =  (df.loc[df['prefecture'] == map_df.loc[i,'prefecture'],datacolumn].values[0]/scalingfactor)
    except IndexError: 
        print(map_df.loc[i,'prefecture'])

import shapely

## If I want to make the map curvey I need to do something like the d3 algorithm (see my d3 notes)

# map_df.loc[map_df['prefecture'] == "沖縄県", 'geometry'] = map_df.loc[map_df['prefecture'] == "沖縄県", 'geometry'].affine_transform([1, 0, 0, 1, 7.5, 14.5])
map_df.loc[map_df['prefecture'] == "沖縄県", 'geometry'] = map_df.loc[map_df['prefecture'] == "沖縄県", 'geometry'].affine_transform([1, 0, 0, 1, 6.5, 13])

rotation_angle = -17
rotation_origin = map_df[map_df.is_valid].unary_union.centroid
map_df['geometry']=map_df['geometry'].rotate(rotation_angle,origin=rotation_origin)
national_map_df['geometry']=national_map_df['geometry'].rotate(rotation_angle,origin=rotation_origin)

title_strings = {'en':r"\textbf{Winning and losing prefectures}" + '\n' + r"\textbf{"+str(mapyear)+"}", 
'jp':r"\textbf{儲かっている、損している都道府県}" + '\n' + r"\textbf{"+str(mapyear)+"}"}#都道府県での勝ち組と負け組
labels = {'en':r'Profit per capita (\textyen/person)', 'jp':r'一人当たりのふるさと納税純利益（円／人）'}
langs = ['en','jp']

for lang in langs:
    fig, ax  = init_plotting(style='map', figsize=(6,5), fontsize=10)
    #fig, ax  = init_plotting(figsize=(6,5), fontsize=10)

    fig.suptitle(title_strings[lang], x=-.18,y=1.0, fontsize=16,ha='left',va='top', transform=ax.transAxes, wrap=True)

    from matplotlib.colors import TwoSlopeNorm
    bins = [-11000,-6000,-1000, 0, 1000, 6000, 11000, 21000]
    bluecolors = matplotlib.cm.get_cmap('coolwarm')(np.linspace(0,.43,3))
    redcolors = matplotlib.cm.get_cmap('coolwarm')(np.linspace(.57,1,4))
    rgbacolors = np.concatenate([bluecolors,redcolors])
    from matplotlib.colors import ListedColormap
    #cmap = ListedColormap([matplotlib.colors.to_hex(color) for color in rgbacolors])
    cmap, norm = matplotlib.colors.from_levels_and_colors(bins, rgbacolors, extend="neither")

    map_df.plot(column=datacolumn, ax=ax,  cmap=cmap, norm=norm,legend=False,lw=.05, edgecolor='#4341417c')#edgecolor='grey'#scheme='userdefined', edgecolor=(0,0,0),lw=.1, classification_kwds={'bins':bins}
    #national_map_df.plot(ax=ax,edgecolor=(0,0,0),facecolor="None",lw=.3)
    #map_df.loc[map_df['prefecture'] == "沖縄県"].plot(ax=ax,edgecolor=(0,0,0),facecolor="None",lw=.3)#scheme='userdefined', edgecolor=(0,0,0),lw=.1, classification_kwds={'bins':bins}

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="2%", pad=0.05)
    cb2 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm,
                                    boundaries=  bins,
                                    extend='neither',
                                    ticks=bins,
                                    #spacing='proportional',
                                    spacing='uniform',
                                    orientation='horizontal')
    from matplotlib.ticker import FuncFormatter,StrMethodFormatter
    cax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))

    cb2.set_label(labels[lang])

    #https://medium.datadriveninvestor.com/creating-a-discrete-colorbar-with-custom-bin-sizes-in-matplotlib-50b0daf8dd46

    # norm = TwoSlopeNorm(vmin=map_df[datacolumn].min(), vcenter=0, vmax=map_df[datacolumn].max())
    # map_df.plot(column=datacolumn, ax=ax, norm=norm, cmap='coolwarm',legend=True)

    #https://stackoverflow.com/questions/36008648/colorbar-on-geopandas
    ax.set_xlim([128.5,147.2])
    ax.set_ylim([33.3,44.2])

    # pc=[135.2,40]
    # L = 2.5
    # phi = 59*np.pi/180
    # theta = 150*np.pi/180
    pc=[135.2,40]
    L = 1.5
    phi = 59*np.pi/180
    theta = 150*np.pi/180
    pR = [L*np.sin(phi)+pc[0],L*np.cos(phi)+pc[1]]
    pL = [pc[0]-L*np.cos(np.pi/2-theta+phi),pc[1]+L*np.sin(np.pi/2-theta+phi)]
    ax.plot([pL[0],pc[0],pR[0]], [pL[1],pc[1],pR[1]], color='grey',lw=1)

    fig.savefig(PLOT_FOLDER+'prefecture-profit_' + lang + '.pdf',bbox_inches="tight",transparent=True)
    fig.savefig(PLOT_FOLDER+'prefecture-profit_' + lang + '.png',bbox_inches="tight",transparent=True)
    fig.savefig(EXTRA_PLOT_FOLDER+'prefecture-profit_' +lang +'.pdf',bbox_inches="tight",transparent=True)
    plt.close('all')

print(len(df.loc[df['profitperperson']<0]))

###########################
### Matplotlib version -- local ###
###########################

mapyear = 2021
map_df = load_map(mapyear,level='local_dc')
df = furusato_df.loc[furusato_df['year']==mapyear]

datacolumn= "netgainminusdeductions"
scalingfactor = hyakuman

map_df[datacolumn] = 0
for i in range(len(map_df)):
    try: 
        map_df.loc[i, datacolumn] =  (df.loc[df['code'] == map_df.loc[i,'code'],datacolumn].values[0]/scalingfactor)
    except IndexError: 
        print(map_df.loc[i,'prefecture'])

fig, ax  = init_plotting()
map_df.plot(column=datacolumn, ax=ax, rasterized=True)
#fig.savefig(PLOT_FOLDER+'profit.pgf')
fig.savefig(PLOT_FOLDER+'profit.pdf', dpi=1000)
## or increase my tex memory using this
## https://tex.stackexchange.com/questions/7953/how-to-expand-texs-main-memory-size-pgfplots-memory-overload
## https://matplotlib.org/stable/tutorials/text/pgf.html
plt.close('all')

##############################
#### PER PERSON PROFIT-LOSS ##
##############################

year = [2016,2021]

title_strings = {'en':r"\textbf{Gap between winners and losers is growing}", 
'jp':r"\textbf{都道府県の間に、利益格差が広がっている}"}
subtitle_strings = {'en':str(year[0]) + r" to " + str(year[1]) + " evolution of population and per-capita profit", 
'jp':str(year[0]) + r"から" + str(year[1])+r"、各県の人口と一人当たり純利益の進化"}
xlabels = {'en':r'Profit per capita  (\textyen/person)', 
'jp':r'一人当たり純利益 (円／人）'} 
ylabels = {'en':'Population (millions)', 
'jp':r'人口 (百万人）'} 
langs = ['en','jp']

for lang in langs: 
    fig, ax = init_plotting(style='nyt')
    ax.set_title(subtitle_strings[lang], x=0., y=1.05, fontsize=14,ha='left',va='bottom',wrap=True)
    fig.suptitle(title_strings[lang], x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes, wrap=True)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabels[lang])
    ax.set_ylabel(ylabels[lang])
    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))

    deltas_jp = {"東京都":[0,-.5], "神奈川県":[200,.5], "大阪府":[0,-.5], "愛知県":[-2200,.2], "埼玉県":[200,-.4], "千葉県":[-2000,0], "兵庫県":[0,-.5], "福岡県":[-100,.5],"北海道":[0,-.5], "静岡県":[0,.5], "茨城県":[900,0], "京都府":[-1100,-.3], "広島県":[0,+.5], "奈良県":[-1700,0],"佐賀県":[5900,-.4],"鹿児島県":[4900,0],"宮崎県":[5900,.5], "山形県":[4000,.5],"熊本県":[1500,.5],"新潟県":[1200,.55], "徳島県":[0,-.5], "鳥取県":[0,-.5], "高知県":[2000,-.5], "山梨県":[7000,-.5]}
    deltas_en = {"東京都":[0,-.5], "神奈川県":[200,.5], "大阪府":[0,-.5], "愛知県":[-2200,.1], "埼玉県":[200,-.4], "千葉県":[-2300,0], "兵庫県":[0,-.5], "福岡県":[-100,.5],"北海道":[0,.5], "静岡県":[0,.5], "茨城県":[1200,0], "京都府":[-200,-.4], "広島県":[0,+.5], "奈良県":[0,-.5],"佐賀県":[5900,-.4],"鹿児島県":[4900,.3],"宮崎県":[6400,.5], "山形県":[4000,.5],"熊本県":[2500,.4],"新潟県":[1800,.4], "徳島県":[0,-.5], "鳥取県":[0,-.5], "高知県":[2000,-.4], "山梨県":[7000,-.5]}
    for prefecture in fn_pop_pref_df['prefecture'].unique(): 

        xs = fn_pop_pref_df.loc[(fn_pop_pref_df['prefecture']==prefecture) &(fn_pop_pref_df['year'].isin(year))]['profitperperson'].values
        ys = fn_pop_pref_df.loc[(fn_pop_pref_df['prefecture']==prefecture) & (fn_pop_pref_df['year'].isin(year))]['total-pop'].values / 10**6
        #ax.plot(fn_pop_pref_df.loc[(fn_pop_pref_df['prefecture']==prefecture) &(fn_pop_pref_df['year'].isin(year))]['profitperperson'],fn_pop_pref_df.loc[(fn_pop_pref_df['prefecture']==prefecture) & (fn_pop_pref_df['year'].isin(year))]['total-pop'])
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
                label = pref_names_df.loc[pref_names_df['prefecture']==prefecture, 'prefecture-reading'].iloc[0]
                label = label.replace('ou','o').replace('oo','o').title()
            ax.text(np.mean(xs)+delta[0],np.mean(ys)+delta[1],label, fontsize=8, ha='center',va='center')
        
        if prefecture == '東京都':
            ax.text(xs[0],ys[0]+.15,year[0], fontsize=8, ha='center',va='bottom')
            ax.text(xs[1]+300,ys[1]-.2,year[1], fontsize=8, ha='center',va='top')

    ax.set_ylim([0,1.41*10])
    ax.set_xlim([-10000,21000])
    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+'population_vs_profitperperson_'+lang+'.'+suffix, transparent=True,bbox_inches="tight")
    plt.close('all')

shutil.copy(PLOT_FOLDER+'population_vs_profitperperson_en.pdf',EXTRA_PLOT_FOLDER)
shutil.copy(PLOT_FOLDER+'population_vs_profitperperson_jp.pdf',EXTRA_PLOT_FOLDER)

import plotly.express as px
import plotly.graph_objects as go


fig = px.line(fn_pop_pref_df.loc[furusato_pref_df['year'].isin(year)], y='total-pop',x='profitperperson', color='prefecture',hover_data=['prefecture', 'year'],markers=True)
# fig = px.scatter(furusato_pref_sum_df, x='donations',y='netgainminusdeductions', color='prefecture',hover_data=['prefecture'])
fig.write_html(PLOT_FOLDER+'/population_vs_profitperperson.html')
plt.close('all')


######################################
### ORDERED PER PERSON PROFIT-LOSS  ##
######################################

# year = 2021

# title_strings = {'en':r"\textbf{Winners and losers}", 
# 'jp':r"\textbf{都道府県の間に、利益格差が広がっている}"}
# subtitle_strings = {'en':r"Ranked profit per capita ", 
# 'jp':str(year) + r"各県の人口と一人当たり純利益の進化"}

# ylabels = {'en':r'Profit per capita (\textyen/person)', 
# 'jp':r'人口 (百万人）'} 
# langs = ['en','jp']

# datacolumn = 'profitperperson'
# #datacolumn = 'netgainminusdeductions'

# for lang in langs: 
#     fig, ax = init_plotting(style='nyt')
#     ax.set_title(subtitle_strings[lang], x=0., y=1.05, fontsize=14,ha='left',va='bottom',wrap=True)
#     fig.suptitle(title_strings[lang], x=0,y=1.2, fontsize=18,ha='left',va='bottom', transform=ax.transAxes, wrap=True)
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

####################################################################
#### PER PERSON PROFIT-LOSS MAP ####################################
####################################################################
mapyear = 2021
map_df = load_map(mapyear,level='prefecture')
df = fn_pop_pref_df.loc[fn_pop_pref_df['year']==mapyear]
#df = furusato_pref_sum_df

datacolumn= "profitperperson"
datacolumnalias = "Profit per person (円/person)"
scalingfactor = 1

map_df[datacolumn] = 0
for i in range(len(map_df)):
    try: 
        map_df.loc[i, datacolumn] =  (df.loc[df['prefecture'] == map_df.loc[i,'prefecture'],datacolumn].values[0]/scalingfactor)
    except IndexError: 
        print(map_df.loc[i,'code'])
        print(map_df.loc[i,'city'])

#largestdeviation = np.max(np.abs(map_df[datacolumn]))
#largestdeviation = 20000

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
        return colormap(feature["properties"][datacolumn]) 
    except ValueError: 
        print(feature["properties"]["name"])
        return 'black'
    except KeyError:
        print(feature["properties"]["name"])
        return 'black'

folium.GeoJson(map_df
,name = datacolumn
,style_function=lambda feature: 
{'fillColor':fillColor(feature)} | unhighlighted_style
,highlight_function=lambda feature: 
{'fillColor':fillColor(feature)} | highlighted_style
,zoom_on_click=True
,tooltip=tooltip
).add_to(m)

colormap.caption = datacolumnalias
try:
    colormap.add_to(m)
except AttributeError:
    pass

m.save(PLOT_FOLDER+"profitperperson_prefecture_map.html")
