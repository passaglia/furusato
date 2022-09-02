import xlwings as xw 
import shutil

CALCULATOR_FILE = './calculator_edit.xlsx'

book = xw.Book(CALCULATOR_FILE)
sheet = book.sheets[0]

incomeCell = "W9"
maxDeductionCell = "W49"
localTaxPreDeductionCell = "W40"
incomeTaxPreDeductionCell = "W78"
totalTaxPaidCell = "W82"
maxIncomeTaxDeductionCell='W90'
maxLocalTaxBaseDeductionCell='W91'
maxLocalTaxSpecialDeductionCell='W92'

def maxDonation(income):
    sheet[incomeCell].value = income
    return sheet[maxDeductionCell].value

def maxDonationBreakdown(income):
    sheet[incomeCell].value = income
    return (sheet[maxIncomeTaxDeductionCell].value,
    sheet[maxLocalTaxBaseDeductionCell].value,
    sheet[maxLocalTaxSpecialDeductionCell].value)

assert (maxDonation(10**7) == 2000+sum(maxDonationBreakdown(10**7)))

import numpy as np
import matplotlib.pyplot as plt
from samplot.circusboy import CircusBoy
from samplot.colors import nice_colors
from matplotlib.ticker import FuncFormatter,StrMethodFormatter

cb = CircusBoy(baseFont=['Helvetica','Hiragino Maru Gothic Pro'],titleFont=['Helvetica','Hiragino Maru Gothic Pro'],textFont=['Helvetica','Hiragino Maru Gothic Pro'], fontsize=12,figsize=(6,4))

save_folder = './calculatorgraph/'
save_folder2 = '../furusato-private/draft/figures/'

man = 10000
incomes = np.linspace(.1*man, 3000*man,200)
deductions = np.array([maxDonation(income)-2000 for income in incomes])
breakdowns = np.array([maxDonationBreakdown(income) for income in incomes])


scalings = {'en': 10**6, 'jp': 10000}
titles = {'en':r"\textbf{High earners can get many more gifts}", 
'jp':r"\textbf{ふるさと納税の逆進性}"}
subtitles ={'en':r"Proportion of income which is eligible for donation", 
'jp':r'給与収入による最大寄付額の割合'}
xlabels = {'en':'Personal income (￥mn)', 
'jp':r'給与収入 （万円）'}

textlabels = {'en':r'(For a household with no dependents)', 'jp':r'（単身世帯の場合）'}
langs = ['en','jp']
for lang in langs: 

    fig, ax = cb.handlers()

    cb.set_titleSubtitle(ax, titles[lang], subtitles[lang])
    cb.set_yTickLabels(ax)
    #cb.set_source(ax, "Data: Ministry of Internal Affairs",loc='outside',pad=10)
    #cb.set_byline(ax, "Sam Passaglia",pad=-4.5)

    ax.set_xlabel(xlabels[lang],color='black')
    ax.plot(incomes/scalings[lang],np.array(deductions/incomes)*100, color=nice_colors(3))
    from matplotlib.offsetbox import AnchoredText
    
    ax.annotate(textlabels[lang], xy=(1, 0), xytext=(-4, +4), va='bottom', ha='right', xycoords='axes fraction', textcoords='offset points',color=cb.grey)# bbox=dict(boxstyle='round', fc='w'),
    #ax.plot(incomes/scalings[lang],np.array(breakdowns[:,0]/incomes)*100, color=nice_colors(2),ls='--')
    #ax.plot(incomes/scalings[lang],np.array((breakdowns[:,1]+breakdowns[:,2])/incomes)*100, color=nice_colors(1))
    ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.0f}\%'.format))
    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.set_ylim([0,4.05])
    ax.set_xlim([0,30*10**6/scalings[lang]])
    #ax.axis('tight')
    fig.savefig(save_folder+'fracdeduction_vs_income_' + lang + '.pdf', bbox_inches='tight',transparent=True)
    fig.savefig(save_folder+'fracdeduction_vs_income_' + lang + '.png')
    plt.close('all')

shutil.copy(save_folder+'fracdeduction_vs_income_en.pdf',save_folder2)
shutil.copy(save_folder+'fracdeduction_vs_income_jp.pdf',save_folder2)

## Regressiveness http://illusttax.com/furusato-tax/
#https://www.mof.go.jp/tax_policy/summary/income/b02.htm

############
### Ratio of Deductions 
############

scalings = {'en': 10**6, 'jp': 10000}
titles = {'en':r"\textbf{Ratio of 住民税控除　to 所得税控除}", 
'jp':r"\textbf{給与収入により、住民税控除/所得税控除の率}"}
subtitles ={'en':r"If you donated the max amount ", 
'jp':r'最大寄附金格の場合。寄附金控除額の計算シミュレーション。'}
xlabels = {'en':'Personal income (￥mn)', 
'jp':r'給与収入 （万円）'}

textlabels = {'en':r'(For a household with no dependents)', 'jp':r'（単身世帯の場合）'}
langs = ['en','jp']
for lang in langs: 

    fig, ax == cb.handlers()
    ax.set_title(subtitles[lang], x=0., y=1.01, fontsize=14,ha='left',va='bottom')
    fig.suptitle(titles[lang], x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
    ax.set_xlabel(xlabels[lang])
    #ax.plot(incomes/scalings[lang],np.array(deductions/incomes)*100, color=nice_colors(3))
    from matplotlib.offsetbox import AnchoredText
    
    ax.annotate(textlabels[lang], xy=(1, 1), xytext=(-12, -12), size=14, va='top', ha='right', xycoords='axes fraction', textcoords='offset points',color='grey')# bbox=dict(boxstyle='round', fc='w'),
    #ax.plot(incomes/scalings[lang],np.array(breakdowns[:,0]/incomes)*100, color=nice_colors(2),ls='--')
    ax.plot(incomes/scalings[lang],np.array((breakdowns[:,1]+breakdowns[:,2])/breakdowns[:,0]), color=nice_colors(1))
    ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.1f}'.format))
    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    #ax.set_ylim([0,4.05])
    ax.axhline(1.9, color='black')
    ax.annotate(r"1.9", xy=(0, 1.9), xytext=(-5, 0), size=12, va='center', ha='right', xycoords='data', textcoords='offset points',color='black')# bbox=dict(boxstyle='round', fc='w'),

    ax.set_xlim([0,30*10**6/scalings[lang]])
    #ax.axis('tight')
    fig.savefig(save_folder+'ratio_vs_income_' + lang + '.pdf', bbox_inches='tight',transparent=True)
    fig.savefig(save_folder+'ratio_vs_income_' + lang + '.png')
    plt.close('all')

# shutil.copy(save_folder+'fracdeduction_vs_income_en.pdf',save_folder2)
# shutil.copy(save_folder+'fracdeduction_vs_income_jp.pdf',save_folder2)

############
### Deductions 
############

scalings = {'en': 10**6, 'jp': 10000}
titles = {'en':r"\textbf{Ratio of 住民税控除　to 所得税控除}", 
'jp':r"\textbf{給与収入により、住民税控除/所得税控除の率}"}
subtitles ={'en':r"If you donated the max amount ", 
'jp':r'最大寄附金格の場合。寄附金控除額の計算シミュレーション。'}
xlabels = {'en':'Personal income (￥mn)', 
'jp':r'給与収入 （万円）'}

textlabels = {'en':r'(For a household with no dependents)', 'jp':r'（単身世帯の場合）'}
langs = ['en','jp']
for lang in langs: 

    fig, ax = cb.handlers()
    ax.set_title(subtitles[lang], x=0., y=1.01, fontsize=14,ha='left',va='bottom')
    fig.suptitle(titles[lang], x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
    ax.set_xlabel(xlabels[lang])
    #ax.plot(incomes/scalings[lang],np.array(deductions/incomes)*100, color=nice_colors(3))
    from matplotlib.offsetbox import AnchoredText
    
    ax.annotate(textlabels[lang], xy=(1, 1), xytext=(-12, -12), size=14, va='top', ha='right', xycoords='axes fraction', textcoords='offset points',color='grey')# bbox=dict(boxstyle='round', fc='w'),
    ax.plot(incomes/scalings[lang],np.array(breakdowns[:,0]), color=nice_colors(2))
    ax.plot(incomes/scalings[lang],np.array((breakdowns[:,1]+breakdowns[:,2])), color=nice_colors(1))

    ax.yaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    ax.xaxis.set_major_formatter(FuncFormatter(r'{0:.0f}'.format))
    #ax.set_ylim([0,4.05])
    ax.axhline((3621*10**8)/(366*10**4), color=nice_colors(1))
    ax.axhline((1904*10**8)/(366*10**4), color=nice_colors(2))
    #ax.annotate(r"1.9", xy=(0, 1.9), xytext=(-5, 0), size=12, va='center', ha='right', xycoords='data', textcoords='offset points',color='black')# bbox=dict(boxstyle='round', fc='w'),

    ax.set_xlim([0,30*10**6/scalings[lang]])
    #ax.axis('tight')
    fig.savefig(save_folder+'deduction_vs_income_' + lang + '.pdf', bbox_inches='tight',transparent=True)
    fig.savefig(save_folder+'deduction_vs_income_' + lang + '.png')
    plt.close('all')

(3621*10**8)/(366*10**4)
(1904*10**8)/(366*10**4)