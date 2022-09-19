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
import samplot.colors as samcolors
import seaborn as sns
import branca.colormap as cm
import plotly.express as px
import plotly.graph_objects as go
import shutil
import jaconv 

from japandata.furusatonouzei.data import furusato_df

import contextily as cx
from xyzservices import TileProvider

from config import mapboxToken, tileCacheDirectory

cx.set_cache_dir(tileCacheDirectory)

def make_pref(PREFECTURE):
    print(PREFECTURE)
    PLOT_FOLDER = os.path.join(os.getcwd(),'singlepref/'+PREFECTURE +'/')
    EXTRA_PLOT_FOLDER = './furusato-private/draft/figures/prefectures'

    output_filetypes = ['pdf']
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    oku = 10**8
    hyakuman = 10**6

    ######################################
    ### Restricting to one prefecture ####
    ######################################

    local_df = fn_pop_ind_df.loc[fn_pop_ind_df['prefecture'] == PREFECTURE]
    pref_df = fn_pop_ind_pref_df.loc[fn_pop_ind_pref_df['prefecture'] == PREFECTURE]

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

    ############################################
    #### Restricting to one year and cleanup ###
    ############################################
    year = 2021
    local_df_year = local_df.loc[local_df['year']==year]
    local_df_year = local_df_year.drop(['city','city_x','city_y'],axis=1)
    local_df_year= local_df_year.reset_index(drop=True)

    # ##############################
    # #### FOLIUM MAP OF DONATIONS ###
    # ##############################

    # datacolumn= "donations"
    # datacolumnalias = "Total Donations (百万円)"
    # scalingfactor = hyakuman

    # df = local_df_with_map.copy()
    # df['dummy'] = df[datacolumn]/scalingfactor

    # largestdeviation = np.max(np.abs(df['dummy']))
    # #largestdeviation = 1000

    # rgbacolors = (matplotlib.cm.get_cmap('coolwarm')(np.linspace(0.5,1,11)))
    # colormap = cm.LinearColormap( [matplotlib.colors.to_hex(color) for color in rgbacolors],
    #     vmin=0, vmax=largestdeviation
    # )

    # center =  df[df.is_valid].unary_union.centroid.coords.xy
    # m = folium.Map(**{'location':[center[1][0], center[0][0]], 'zoom_start':9, 'tiles':'None','attr':" "})

    # tooltip = GeoJsonTooltip(
    #     fields=["prefecture", "city", "dummy"],
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
    #         return colormap(feature["properties"]['dummy']) 
    #     except ValueError: 
    #         print(feature["properties"]["name"])
    #         return 'black'
    #     except KeyError:
    #         print(feature["properties"]["name"])
    #         return 'black'

    # folium.GeoJson(df
    # ,name = 'dummy'
    # ,style_function=lambda feature: 
    # {'fillColor':fillColor(feature)} | unhighlighted_style
    # ,highlight_function=lambda feature: 
    # {'fillColor':fillColor(feature)} | highlighted_style
    # ,zoom_on_click=True
    # ,tooltip=tooltip
    # ).add_to(m)

    # colormap.caption = datacolumnalias
    # colormap.add_to(m)

    # m.save(PLOT_FOLDER+PREFECTURE+"donations_map.html")

    # ##############################
    # #### FOLIUM MAP OF PROFIT ###
    # ##############################

    # datacolumn= "netgainminusdeductions"
    # datacolumnalias = "Net Gain Minus Deductions (百万円)"
    # scalingfactor = hyakuman

    # df = local_df_with_map.copy()
    # df['dummy'] = df[datacolumn]/scalingfactor

    # largestdeviation = np.max(np.abs(df['dummy']))
    # #largestdeviation = 2000

    # rgbacolors = (matplotlib.cm.get_cmap('coolwarm')(np.linspace(0,1,11)))
    # colormap = cm.LinearColormap( [matplotlib.colors.to_hex(color) for color in rgbacolors],
    #     vmin=-largestdeviation, vmax=largestdeviation
    # )

    # center =  df[df.is_valid].unary_union.centroid.coords.xy
    # m = folium.Map(**{'location':[center[1][0], center[0][0]], 'zoom_start':9, 'tiles':'None','attr':" "})

    # tooltip = GeoJsonTooltip(
    #     fields=["prefecture", "city", "city-reading", "code", "total-pop",  "donations", "donations-fraction", "economic-strength-index", "dummy"],
    #     aliases=["Prefecture:", "City:", "City (en):","Code:", "Population:", "Donations:", "Fraction of all donations in Prefecture:", "Economic Strength Index", datacolumnalias + ':'],
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
    #         return colormap(feature["properties"]["dummy"]) 
    #     except ValueError: 
    #         print(feature["properties"]["name"])
    #         return 'black'
    #     except KeyError:
    #         print(feature["properties"]["name"])
    #         return 'black'

    # folium.GeoJson(df
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

    # m.save(PLOT_FOLDER+PREFECTURE+"profit_map.html")

    ##############################
    #### PROFIT VS STRENGTH   ####
    ##############################

    ## TODO: MAKE THIS LESS SHIT AND DO DONATIONS TOO

    fig, ax = init_plotting(style='nyt')
    title_string = r""
    #subtitle_string = r"(1m people)"
    #ax.set_title(subtitle_string, x=0., y=1.0, fontsize=14,ha='left',va='bottom')
    fig.suptitle(title_string, x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
    years = [2019]
    #ax.scatter(local_df.loc[local_df['year'].isin(year)]['economic-strength-index'],local_df.loc[local_df['year'].isin(year)]['profitperperson'])
    #ax.scatter(local_df.loc[local_df['year'].isin(year)]['economic-strength-index'],local_df.loc[local_df['year'].isin(year)]['netgainminusdeductions'])
    ax.scatter(local_df.loc[local_df['year'].isin(years)]['economic-strength-index'],local_df.loc[local_df['year'].isin(years)]['donations-fraction'])
    #ax.scatter(local_df.loc[local_df['year'].isin(year)]['economic-strength-index'],local_df.loc[local_df['year'].isin(year)]['donations'])

    #ax.set_xlim([.2,1])
    #ax.set_ylim([-5000,20000])
    #ax.axhline(0,color='black', alpha=.5, ls='--')
    for suffix in output_filetypes:
        fig.savefig(PLOT_FOLDER+PREFECTURE+'donationsfrac_vs_strength.'+suffix, transparent=True)
    plt.close('all')

    # fig = px.scatter(local_df.loc[local_df['year'].isin(year)], x='economic-strength-index',y='donations', color='prefecture', hover_data=['prefecture'])
    # fig.write_html(PLOT_FOLDER+PREFECTURE+'profitperperson_vs_strength_pref.html')
    # plt.close('all')

    ##################################
    ### Matplotlib version ###########
    ##################################

    map_df = load_map(year,level='local_dc', quality='medium')
    map_df=map_df.loc[map_df['prefecture']==PREFECTURE]
    map_df = map_df.reset_index(drop=True)
    map_df = map_df.drop(map_df.loc[~map_df['code'].isin(local_df_year['code'])].index,axis=0)
    if PREFECTURE == '東京都':
        codes_to_drop = ['13421','13361','13362','13363','13364','13381','13382','13401','13402']
        map_df = map_df.drop(map_df.loc[map_df['code'].isin(codes_to_drop)].index)
    map_df = map_df.reset_index(drop=True)
    
    pref_map_df = load_map(year,level='prefecture', quality='medium')
    pref_map_df=pref_map_df.loc[pref_map_df['prefecture']==PREFECTURE]
    pref_map_df=pref_map_df.reset_index(drop=True)

    local_df_with_map = pd.merge(map_df, local_df_year, on=["code", "prefecture"],validate='one_to_one')
    
    df = local_df_with_map.copy()
    datacolumn= "netgainminusdeductions"
    maxval = df[datacolumn].max()/oku
    if maxval <= 5:
        maxbin = np.ceil(maxval)
        bins = np.arange(0,maxbin+1,1)
    elif 5 <= maxval <= 10:
        bins = [0, 1, 5, 10]
    elif maxval < 60:
        maxbin = np.ceil(maxval/10)*10
        bins = [0, 1, 5] + list(np.arange(10,maxbin+10,10))
    else:
        maxbin = np.ceil(maxval/10)*10
        bins = np.ones(11)
        step = 10
        while len(bins)>7:
            bins = [0,1, 5]+list(np.arange(10,maxbin+10,step))
            if bins[-1]< maxbin:
                bins[-1]=maxbin
            step +=10

    if maxval>1:
        bins[-1] = np.ceil(maxval)
    else:
        bins[-1] = np.ceil(maxval*100)/100
    bins = np.array(bins)*oku

    largenegativenumber = -1000*oku

    print('Total number of municipalities:', len(df[datacolumn]))
    hist, histbins = np.histogram(df[datacolumn], np.array([largenegativenumber]+list(bins)))
    assert (hist.sum() == len(df[datacolumn]))
    for i in range(len(histbins)-1):
        print(str(hist[i]) + " municipalities ( " + str(round(hist[i]/hist.sum()*100)) +" \% ) in range " +  str(histbins[i]/oku)+'-'+ str(histbins[i+1]/oku), 'oku yen' )
        print(df.loc[(histbins[i] < df[datacolumn]) & (df[datacolumn]< histbins[i+1]), ['city-reading','total-pop', 'netgainminusdeductions', 'donations', 'donations-fraction','economic-strength-index']].sort_values('netgainminusdeductions'))

    scalingfactors = {'en':hyakuman, 'jp':oku}

    PREFECTURE_EN = (pref_names_df.loc[pref_names_df['prefecture']==PREFECTURE, 'prefecture-reading'].iloc[0]).replace('ou','o').replace('oo','o').title() 
    
    langs = ['en', 'jp']
    title_strings = {'en':r"\textbf{Furusato n\={o}zei profit in "+ PREFECTURE_EN +  r"  in " + str(year) + "}", 
    'jp': r"\textbf{" + PREFECTURE + "におけるふるさと納税純利益（" + str(year) + "年）}"} #jaconv.h2z(str(year),digit=True)
    xlabels = {'en': r'Profit (\textyen mn)', 'jp': r'純利益 (億円)'}
    losslabels = {'en':r"Loss",'jp':r"損失"}
    labelproviders = {'en':TileProvider({
            'name':'Mapbox Labels Only En',
            'attribution':'',
            'url':'https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}{r}?access_token={accessToken}',
            'accessToken': mapboxToken,
            'id': 'mithridatesvi/cl4xdjlvd000315myqsnkyhn7',
            'r':'@2x'
        }), 'jp': TileProvider({
            'name':'Mapbox Labels Only Jp',
            'attribution':'',
            'url':'https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}{r}?access_token={accessToken}',
            'accessToken': mapboxToken,
            'id': 'mithridatesvi/cl64uc0sx005114s475i57a01',
            'r':'@2x'
        })}

    for lang in langs:

        df = local_df_with_map.copy()
        df['dummy'] = df[datacolumn]/scalingfactors[lang]

        dpi=400
        if PREFECTURE == '北海道':
            dpi=300
        
        bp = BasePlot(figsize=(6,6), fontsize=12,dpi=dpi)
        fig, ax  = bp.handlers()
        plt.axis('off')
        #fig, ax  = init_plotting(style='map', figsize=(6,6), fontsize=10, dpi=dpi)
        fig.suptitle(title_strings[lang], x=0,y=1.01, fontsize=bp.titlesize,ha='left',va='bottom', transform=ax.transAxes, wrap=True)
        #t = fig.suptitle(title_strings[lang], x=0.05,y=.95, fontsize=16,ha='left',va='top', transform=ax.transAxes, wrap=True,bbox=dict(facecolor='white', alpha=1, edgecolor='white'))
        #, boxstyle='round,pad=.1'

        if PREFECTURE != '北海道':
            pref_map_df.plot(ax=ax,edgecolor=(0,0,0),facecolor="None",lw=.3)

        from matplotlib.colors import ListedColormap,Normalize,from_levels_and_colors

        blue = samcolors.nice_colors(3)
        blue = np.array(list(blue)+[1])
        redcolors = matplotlib.cm.get_cmap('coolwarm')(np.linspace(0.5,1,len(bins)-1))
        colors = [blue] + list(redcolors)
        alpha = 1
        for i in range(len(colors)):
            colors[i][-1] = alpha

        cmap, norm = from_levels_and_colors(np.array([largenegativenumber]+list(bins))/scalingfactors[lang], colors, extend="neither")
        df.to_crs(epsg=3857).plot(column='dummy', ax=ax,cmap=cmap, norm=norm, legend=False, edgecolor=(0,0,0),lw=.1)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="2%", pad=0.2)
        cb2 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                        norm=norm,
                                        boundaries=  np.array([-oku]+list(bins))/scalingfactors[lang],
                                        extend='neither',
                                        ticks=bins/scalingfactors[lang],
                                        #spacing='proportional',
                                        spacing='uniform',
                                        orientation='horizontal')
        from matplotlib.ticker import FuncFormatter,StrMethodFormatter
        cax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
        cax.text(largenegativenumber/scalingfactors[lang]/2,1.5, losslabels[lang], ha='center')
        cb2.ax.tick_params(axis='both', colors='none')
        cb2.outline.set_edgecolor('none')
        cb2.set_label(xlabels[lang])

        mainprovider = cx.providers.CartoDB.VoyagerNoLabels(r="@2x")
        
        mainprovider = TileProvider({
            'name':'Mapbox Background Only',
            'attribution':'',
            'url':'https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}{r}?access_token={accessToken}',
            'accessToken': mapboxToken,
            'id': 'mithridatesvi/cl4gf57r8000614mc0xrwsvt7',
            'r':'@2x'
        })

        labelprovider = labelproviders[lang]
            
        xBuffer = .5
        yBuffer = .5
        boundsarr = df[df.is_valid].unary_union.bounds
        center =  df[df.is_valid].unary_union.centroid.coords.xy
        yL = boundsarr[3]- boundsarr[1]

        reaspect = True
        desired_aspect_ratio = 4/3
        if reaspect:
            boundsarr = [center[0][0]-yL/2*desired_aspect_ratio, boundsarr[1], center[0][0]+yL/2*desired_aspect_ratio, boundsarr[3]]

        xL =  boundsarr[2]- boundsarr[0]

        suggested_zoom = cx.tile._calculate_zoom(boundsarr[0]-xBuffer*xL,
                                            boundsarr[1]-yBuffer*yL,
                                            boundsarr[2]+xBuffer*xL,
                                            boundsarr[3]+yBuffer*yL)

        labels_img, labels_ext = cx.bounds2img(*df[df.is_valid].unary_union.bounds,
                                            ll=True,
                                            zoom=suggested_zoom-1,
                                            source=labelprovider
                                            )

        ax.imshow(labels_img, extent=labels_ext,zorder=+10,interpolation='sinc')

        img, ext = cx.bounds2img(boundsarr[0]-xBuffer*xL,
                                boundsarr[1]-yBuffer*yL,
                                boundsarr[2]+xBuffer*xL,
                                boundsarr[3]+yBuffer*yL,
                                ll=True,
                                source=mainprovider,
                                zoom=suggested_zoom
                                )
        ax.imshow(img, extent=ext)

        xBuffer = .02
        yBuffer = .01
        
        boundsarr = pref_map_df.to_crs(epsg=3857).bounds.values[0]
        center =  pref_map_df.to_crs(epsg=3857).centroid[0].coords.xy

        if PREFECTURE in ['東京都','北海道']:
            boundsarr = df[df.is_valid].to_crs(epsg=3857).unary_union.bounds
            center =  df[df.is_valid].to_crs(epsg=3857).unary_union.centroid.coords.xy

        yL = boundsarr[3]- boundsarr[1]

        if reaspect:
            boundsarr = [center[0][0]-yL/2*desired_aspect_ratio, boundsarr[1], center[0][0]+yL/2*desired_aspect_ratio, boundsarr[3]]

        xL =  boundsarr[2]- boundsarr[0]

        ax.set_xlim([boundsarr[0]-xBuffer*xL, boundsarr[2]+xBuffer*xL])
        ax.set_ylim([boundsarr[1]-yBuffer*yL, boundsarr[3]+yBuffer*yL])

        fig.savefig(PLOT_FOLDER+PREFECTURE_EN+'-profit-map_'+lang+'.pdf',bbox_inches="tight")
        fig.savefig(PLOT_FOLDER+PREFECTURE_EN+'-profit-map_'+lang+'.png',bbox_inches="tight")
        plt.close('all')

        os.makedirs(EXTRA_PLOT_FOLDER,exist_ok=True)

        shutil.copy(PLOT_FOLDER+PREFECTURE_EN+'-profit-map_'+lang+'.pdf',EXTRA_PLOT_FOLDER)

        plt.close('all')



# PREFECTURE = "佐賀県"
# PREFECTURE = "秋田県"
# PREFECTURE = "徳島県"
# PREFECTURE = "山形県"
# PREFECTURE = "北海道"
# PREFECTURE = "福岡県"
# PREFECTURE = "千葉県"
# PREFECTURE = "青森県"
# PREFECTURE = "埼玉県"
# PREFECTURE = "東京都"
# PREFECTURE = "北海道"
# PREFECTURE = '奈良県'
# PREFECTURE = '大阪府'
# PREFECTURE="静岡県"
# PREFECTURE="兵庫県"
# make_pref(PREFECTURE)

for PREFECTURE in ['北海道','秋田県']:
    make_pref(PREFECTURE)

# for PREFECTURE in furusato_df["prefecture"].unique():
#     print(PREFECTURE)
#     make_pref(PREFECTURE)

