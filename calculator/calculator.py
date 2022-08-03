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
from samplot.utils import init_plotting
from samplot.colors import nice_colors
from matplotlib.ticker import FuncFormatter,StrMethodFormatter

save_folder = './calculatorgraph/'
save_folder2 = '../furusato-private/draft/figures/'

man = 10000
incomes = np.linspace(.1*man, 3000*man,200)
deductions = np.array([maxDonation(income)-2000 for income in incomes])
breakdowns = np.array([maxDonationBreakdown(income) for income in incomes])
fig, ax = init_plotting(style='nyt')
ax.plot(incomes,deductions, color=nice_colors(2))
fig.savefig(save_folder+'deduction_vs_income.pdf')
plt.close('all')

scalings = {'en': 10**6, 'jp': 10000}
title_strings = {'en':r"\textbf{High earners can get many more gifts}", 
'jp':r"\textbf{ふるさと納税の逆進性}"}
subtitle_strings ={'en':r"Proportion of income which is eligible for donation", 
'jp':r'給与収入による最大寄付額の割合'}
xlabels = {'en':'Personal income (￥mn)', 
'jp':r'給与収入 （万円）'}

textlabels = {'en':r'(For a household with no dependents)', 'jp':r'（単身世帯の場合）'}
langs = ['en','jp']
for lang in langs: 

    fig, ax = init_plotting(style='nyt')
    ax.set_title(subtitle_strings[lang], x=0., y=1.01, fontsize=14,ha='left',va='bottom')
    fig.suptitle(title_strings[lang], x=0,y=1.15, fontsize=18,ha='left',va='bottom', transform=ax.transAxes)
    ax.set_xlabel(xlabels[lang])
    ax.plot(incomes/scalings[lang],np.array(deductions/incomes)*100, color=nice_colors(3))
    from matplotlib.offsetbox import AnchoredText
    
    ax.annotate(textlabels[lang], xy=(1, 0), xytext=(-12, +12), size=14, va='bottom', ha='right', xycoords='axes fraction', textcoords='offset points',color='grey')# bbox=dict(boxstyle='round', fc='w'),
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