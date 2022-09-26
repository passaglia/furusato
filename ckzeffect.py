import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from samplot.circusboy import CircusBoy
import samplot.colors as samcolors
import shutil
from matplotlib.ticker import FuncFormatter

from data import furusato_df, furusato_rough_df, annual_df, rough_annual_df

from config import productionPlotFolder

PLOT_FOLDER = os.path.join(os.path.dirname(__file__),'ckzeffect/')

output_filetypes = ['pdf','png']

os.makedirs(PLOT_FOLDER, exist_ok=True)
#os.makedirs(productionPlotFolder, exist_ok=True)

for filetype in output_filetypes:
    os.makedirs(PLOT_FOLDER+'/'+filetype, exist_ok=True)

oku = 10**8

cb = CircusBoy(baseFont='Helvetica',cjkFont='Hiragino Maru Gothic Pro',titleFont='Helvetica',textFont='Helvetica',fontsize=12,figsize=(6,4))

#### Prefecture ckz effect

furusato_df.loc[(furusato_df['city']=='prefecture') & (furusato_df['year']==2020), ['prefecture','year','deductions','profit-from-ckz']]

assert(np.abs(furusato_df.loc[(furusato_df['city']=='prefecture') & (furusato_df['year']==2020), ['profit-from-ckz']].sum().values[0])<1000)

## Why is tokyo's profit from CKZ so negative in 2019? It should get 0 ckz either way. This was because I didnt account for the exception in how tokyo is computed: basically it can never get ckz
furusato_df.loc[(furusato_df['city']=='prefecture') & (furusato_df['prefecture'] == '東京都'), ['prefecture','year','profit-from-ckz']]

##

# topN = 10
# titles = {'en':r"\textbf{Municipalities which receive the most donations}",'jp':r"\textbf{ふるさと納税受入額の多い10自治体}"}

# langs = ['en','jp']
# for lang in langs:
#     if lang == 'en':
#         colname = 'city-reading'
#     elif lang =='jp':
#         colname = 'city'

#     cbtable = CircusBoy(baseFont='Helvetica', cjkFont='Hiragino Maru Gothic Pro',titleFont='Helvetica',textFont='Helvetica', fontsize=12,figsize=(6.75,4))
#     fig, ax = cbtable.handlers()
#     fig.suptitle(titles[lang], y=.92, fontsize=cbtable.titlesize)
#     code_list = []
#     names_list = [] 
#     for i, year in enumerate(annual_df.year):
#         inds = np.argsort(furusato_df.loc[furusato_df['year']==year]['donations'].values)[::-1]
#         code_list.append(furusato_df.loc[furusato_df['year']==year]['code'].iloc[inds][0:topN].values)
#         names_list.append([s.title().replace('ou', r'\={o}') for s in furusato_df.loc[furusato_df['year']==year][colname].iloc[inds][0:topN].values])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     plt.box(on=None)
#     uniques,counts=np.unique(code_list,return_counts=True)
#     colors = dict()
#     totalcolorsneeded = len(np.where(counts>1)[0])
#     j=0
#     cmap=matplotlib.cm.get_cmap('tab20c')
#     for i, unique in enumerate(uniques):
#         if counts[i] == 1:
#             colors[unique] = "w"
#         else:
#             colors[unique] = cmap(j/(totalcolorsneeded)*16/20)
#             j+=1

#     code_list = np.array(code_list).T
#     names_list = np.array(names_list).T
#     cellColours = np.ones_like(code_list)
#     for i in range(len(code_list)):
#         for j in range(len(code_list[i])):
#             cellColours[i,j] = colors[code_list[i,j]]
        
#     table = ax.table(cellText=names_list,
#             cellColours=cellColours.tolist(),
#             colLabels=annual_df.year,
#             rowLabels=[str(i+1) + r'.' for i in range(topN)],
#             cellLoc='center',
#             loc='center',
#             fontsize=10.5)
#     table.auto_set_font_size(False)
#     table.set_fontsize(10.5)
#     table.scale(1.1, 2)

#     import six
#     for k, cell in six.iteritems(table._cells):
#         cell.set_edgecolor('white')
#         if k[1] == -1:
#             cell.set_text_props(ha='center')
#         # if k[0] == 0 or k[1] < header_columns:
#         #     cell.set_text_props(weight='bold', color='w')
#         #     cell.set_facecolor(header_color)
#         # else:
#         #     cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

#     ax.axis('tight')
#     ax.axis('off')
#     for suffix in output_filetypes:
#         fig.savefig(PLOT_FOLDER+'/'+suffix+'/'+'topN_table_'+lang+'.'+suffix, transparent=True,bbox_inches='tight')
#     plt.close('all')

#     shutil.copy(PLOT_FOLDER+'pdf/topN_table_'+lang+'.pdf',productionPlotFolder)
