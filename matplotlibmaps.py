
##################################
### Matplotlib version -- pref ###
##################################

#national_map_df = load_map(mapyear,level='japan', quality='stylized')
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

map_df.loc[map_df['prefecture'] == "沖縄県", 'geometry'] = map_df.loc[map_df['prefecture'] == "沖縄県", 'geometry'].affine_transform([1, 0, 0, 1, 6.5, 13])

rotation_angle = -17
rotation_origin = map_df[map_df.is_valid].unary_union.centroid
map_df['geometry']=map_df['geometry'].rotate(rotation_angle,origin=rotation_origin)
#national_map_df['geometry']=national_map_df['geometry'].rotate(rotation_angle,origin=rotation_origin)

#map_df = map_df.to_crs("EPSG:3395")
map_df = map_df.to_crs("EPSG:30166")

titles = {'en':r"\textbf{Winning and losing prefectures}" + '\n' + r"\textbf{"+str(mapyear)+"}", 
'jp':r"\textbf{儲かっている、損している都道府県}" + '\n' + r"\textbf{"+str(mapyear)+"}"}#都道府県での勝ち組と負け組
labels = {'en':r'Profit per capita (\textyen/person)', 'jp':r'一人当たりのふるさと納税純利益（円／人）'}
langs = ['en','jp']

for lang in langs:
    bp = BasePlot(figsize=(6,6), fontsize=12)
    fig, ax  = bp.handlers()
    plt.axis('off')

    from matplotlib.colors import TwoSlopeNorm
    bins = [-11000,-6000,-1000, 0, 1000, 6000, 11000, 21000]
    bluecolors = matplotlib.cm.get_cmap('coolwarm')(np.linspace(0,.43,3))
    redcolors = matplotlib.cm.get_cmap('coolwarm')(np.linspace(.57,1,4))
    rgbacolors = np.concatenate([bluecolors,redcolors])
    from matplotlib.colors import ListedColormap
    #cmap = ListedColormap([matplotlib.colors.to_hex(color) for color in rgbacolors])
    cmap, norm = matplotlib.colors.from_levels_and_colors(bins, rgbacolors, extend="neither")

    ax = map_df.plot(column=datacolumn, ax=ax,  cmap=cmap, norm=norm,legend=False,lw=.05, edgecolor='#4341417c')#edgecolor='grey'#scheme='userdefined', edgecolor=(0,0,0),lw=.1, classification_kwds={'bins':bins}
    #national_map_df.plot(ax=ax,edgecolor=(0,0,0),facecolor="None",lw=.3)
    #map_df.loc[map_df['prefecture'] == "沖縄県"].plot(ax=ax,edgecolor=(0,0,0),facecolor="None",lw=.3)#scheme='userdefined', edgecolor=(0,0,0),lw=.1, classification_kwds={'bins':bins}
    ax.set_xlim([-.75*10**6,.98*10**6])
    ax.set_ylim([-.28*10**6,.95*10**6])

    fig.suptitle(titles[lang], x=0,y=1.0, fontsize=bp.titlesize,ha='left',va='top', transform=ax.transAxes)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    #cax = divider.append_axes("bottom", size="2%", pad=0.03)

    cbwidth = ax.get_position().width*.75
    axwidth = ax.get_position().width
    xstart = ax.get_position().x0 + (axwidth-cbwidth)/2
    cbheight = .015
    cax = fig.add_axes([xstart,ax.get_position().y0-cbheight,cbwidth,cbheight])

    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm,
                                    boundaries=bins,
                                    extend='neither',
                                    ticks=bins,
                                    #spacing='proportional',
                                    spacing='uniform',
                                    orientation='horizontal')
    from matplotlib.ticker import FuncFormatter,StrMethodFormatter
    cax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    cb.ax.tick_params(axis='both', colors='none')
    cb.outline.set_edgecolor('none')
    cb.set_label(labels[lang],color='black')
    #cax.xaxis.set_label_position('top')


    #https://medium.datadriveninvestor.com/creating-a-discrete-colorbar-with-custom-bin-sizes-in-matplotlib-50b0daf8dd46

    # norm = TwoSlopeNorm(vmin=map_df[datacolumn].min(), vcenter=0, vmax=map_df[datacolumn].max())
    # map_df.plot(column=datacolumn, ax=ax, norm=norm, cmap='coolwarm',legend=True)

    #https://stackoverflow.com/questions/36008648/colorbar-on-geopandas

    pc=[-1*10**4,4.2*10**5]
    L = 1.2*10**5
    phi = 35*np.pi/180 ## angle to vertical
    theta = 140*np.pi/180 ## opening angle
    pR = [L*np.sin(phi)+pc[0],L*np.cos(phi)+pc[1]]
    pL = [pc[0]-L*np.cos(np.pi/2-theta+phi),pc[1]+L*np.sin(np.pi/2-theta+phi)]
    ax.plot([pL[0],pc[0],pR[0]], [pL[1],pc[1],pR[1]], color='black',lw=1)

    #bp.set_byline(cax, "Sam Passaglia",pad=-4.5)

    fig.savefig(PLOT_FOLDER+'prefecture-profit_' + lang + '.pdf',transparent=True,bbox_inches="tight")
    fig.savefig(PLOT_FOLDER+'prefecture-profit_' + lang + '.png',transparent=True,bbox_inches="tight")
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
