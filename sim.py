import pandas as pd
import numpy as np
from japandata.furusatonouzei.data import furusato_df as fdf
from japandata.chihoukoufuzei.data import local_df as cdf
from japandata.chihoukoufuzei.data import pref_df as cdf_pref

from japandata.indices.data import local_ind_df 
from japandata.indices.data import pref_ind_df 
import matplotlib.pyplot as plt

from samplot.baseplot import BasePlot
from samplot.circusboy import CircusBoy

## Constraints:
## 1 - (Before implementing the seppan funding): Total ckz should be the same
## 2 - If the total FN amount is zero, total debt amount (and everything else) should be the same
## 3 - (Before implementing the debt rollover): Only touch the special-debt, final-demand, ckz
## 4 - final-demand + special-debt = demand-pre-debt
## 5 - final-demand - income should be nearly equal to ckz (up the small adjustment factor and ignoring fukoufudantai)
## 6 - The formula for debt is nominally constant * (yeardf['demand-pre-debt']-yeardf['income']) * economic-strength-index-prev3yearavg with the constant chosen to that the total bond amount leads to a ckz of the desired size.

### seppan data from the soumushou graph
chou=10**12
shortfall = {2022: 0*chou, 2021:1.7*2*chou, 2020:0*chou, 2019:0*chou, 2018:.2*2*chou, 2017:.7*2*chou, 2016:.3*2*chou, 2015:1.5*2*chou}

cb = CircusBoy()
###### LOCAL SIM 
output_df = pd.DataFrame()
simStartYear = np.min(fdf.year)
simEndYear =  np.max(cdf.year)-1
for year in range(simStartYear, simEndYear+1):
    chihoukoufuzeiyear = year+1 #TODO: ACTUALLY THIS SHOULD BE +2. Donations are made in year X, decline in income is reported in year X+1, and then compensation occurs in X+2年度. Implement an estimation of the effect by using the last year's average or something...
    print(year)
    fdfyear = fdf.loc[fdf.year == year]
    ## get rid of the prefectures 
    fdfyear = fdfyear.drop(fdfyear.loc[fdfyear['city']=='prefecture'].index)
    # get rid of the tokyo23
    fdfyear = fdfyear.drop(fdfyear.loc[(fdfyear['prefecture']=='東京都') & (fdfyear['city']).str.contains('区')].index)
    cdfyear = cdf.loc[cdf.year == chihoukoufuzeiyear]
    cdfyear = cdfyear.drop(cdfyear.loc[(cdfyear['prefecture']=='東京都') & (cdfyear['city']).str.contains('区')].index)
    assert(len(cdfyear) == len(fdfyear))

    simulatedcdfyear = cdfyear.copy()
    simulatedcdfyear = simulatedcdfyear.merge(fdfyear,on=['prefecture','code'],validate='one_to_one', suffixes=['','_fdf'])
    simulatedcdfyear['ckz-preAdj'] = simulatedcdfyear['final-demand']-simulatedcdfyear['income'] ## Amount of ckz before final adjustment
    simulatedcdfyear.loc[simulatedcdfyear['ckz-preAdj']<0,'ckz-preAdj'] = 0 ## ckz cant be negative

    simulatedcdfyear.loc[simulatedcdfyear['economic-strength-index-prev3yearavg'].isna(),'economic-strength-index-prev3yearavg'] = simulatedcdfyear['economic-strength-index'] ## For places without a 3-year ESI, use current year ESI

    ## The debt scaling constant that was used for each city in practice
    simulatedcdfyear['debt-scaling-factor'] = simulatedcdfyear['special-debt']/(simulatedcdfyear['demand-pre-debt']-simulatedcdfyear['income'])/simulatedcdfyear['economic-strength-index-prev3yearavg']
    # The adjustment factor that was used to reduce the final demand and get the ckz
    simulatedcdfyear['adjustment-factor'] =  1-(simulatedcdfyear['ckz']+simulatedcdfyear['income'])/simulatedcdfyear['final-demand']
    simulatedcdfyear.loc[simulatedcdfyear['ckz']==0,'adjustment-factor'] = 0

    ## Now to undo the FN effect add back 0.75*deductions
    simulatedcdfyear['income-noFN'] = simulatedcdfyear['income']+0.75*simulatedcdfyear['deductions']
    ## With this higher income, recompute how much debt they would have been allowed to issue

    if shortfall[chihoukoufuzeiyear]>0:
        subsidyFraction = 0.5
    else:
        subsidyFraction = 0
    print('Using subsidy fraction', subsidyFraction)
    governmentSubsidyTotal = subsidyFraction* 0.75*simulatedcdfyear['deductions'].sum() ## For testing purposes

    ## Find a uniform scaling of the debt-scaling-factors such that with the nouzei undone we still get the right total ckz-preAdj
    from scipy.optimize import minimize
    def ckzPreAdjConstraint(extraFactor):
        simulatedcdfyear['special-debt-noFN'] =  (extraFactor*simulatedcdfyear['debt-scaling-factor']*(simulatedcdfyear['demand-pre-debt']-simulatedcdfyear['income-noFN'])*simulatedcdfyear['economic-strength-index-prev3yearavg'])
        simulatedcdfyear['final-demand-noFN'] = simulatedcdfyear['demand-pre-debt']-simulatedcdfyear['special-debt-noFN']
        simulatedcdfyear['ckz-preAdj-noFN'] = simulatedcdfyear['final-demand-noFN']-simulatedcdfyear['income-noFN'] 
        simulatedcdfyear.loc[simulatedcdfyear['ckz-preAdj-noFN']<0,'ckz-preAdj-noFN'] = 0
        return np.abs(simulatedcdfyear['ckz-preAdj-noFN'].sum() - (simulatedcdfyear['ckz-preAdj'].sum() - governmentSubsidyTotal))

    res = minimize(ckzPreAdjConstraint, 1)
    extraDebtFactorFound = res.x
    simulatedcdfyear['debt-scaling-factor-noFN'] = extraDebtFactorFound*simulatedcdfyear['debt-scaling-factor']
    simulatedcdfyear['special-debt-noFN'] =  simulatedcdfyear['debt-scaling-factor-noFN']*(simulatedcdfyear['demand-pre-debt']-simulatedcdfyear['income-noFN'])*simulatedcdfyear['economic-strength-index-prev3yearavg']
    simulatedcdfyear['final-demand-noFN'] = simulatedcdfyear['demand-pre-debt']-simulatedcdfyear['special-debt-noFN']
    simulatedcdfyear['ckz-preAdj-noFN'] = simulatedcdfyear['final-demand-noFN']-simulatedcdfyear['income-noFN'] 
    simulatedcdfyear.loc[simulatedcdfyear['ckz-preAdj-noFN']<0,'ckz-preAdj-noFN'] = 0
    #assert(np.abs(1-simulatedcdfyear['ckz-preAdj-noFN'].sum()/simulatedcdfyear['ckz-preAdj'].sum())<0.001)

    ## Now do the same for the adjustment factor.
    ## Find a uniform scaling of the adjustment-factors such that with the nouzei undone we still get the right total ckz
    def ckzConstraint(extraFactor):
        simulatedcdfyear['ckz-noFN'] = simulatedcdfyear['final-demand-noFN'] * (1- extraFactor * simulatedcdfyear['adjustment-factor']) - simulatedcdfyear['income-noFN']
        simulatedcdfyear.loc[simulatedcdfyear['ckz-noFN']<0,'ckz-noFN'] = 0
        return np.abs(simulatedcdfyear['ckz-noFN'].sum()-(simulatedcdfyear['ckz'].sum()-governmentSubsidyTotal)) 

    res = minimize(ckzConstraint, 1)
    extraAdjFactorFound = res.x
    simulatedcdfyear['adjustment-factor-noFN'] = extraAdjFactorFound*simulatedcdfyear['adjustment-factor']
    simulatedcdfyear['ckz-noFN'] = simulatedcdfyear['final-demand-noFN'] * (1- simulatedcdfyear['adjustment-factor-noFN']) - simulatedcdfyear['income-noFN']
    simulatedcdfyear.loc[simulatedcdfyear['ckz-noFN']<0,'ckz-noFN'] = 0
    #assert(np.abs(1-simulatedcdfyear['ckz-noFN'].sum()/simulatedcdfyear['ckz'].sum())<0.001)
    
    simulatedcdfyear['ckz-noFN'].sum()/simulatedcdfyear['ckz'].sum()

    # In principle if the money is subsidized most places should get more ckz with FN than without, since FN decreases their income.
    myFNeffect = (simulatedcdfyear['ckz']-simulatedcdfyear['ckz-noFN'])

    print(len(simulatedcdfyear.loc[myFNeffect>0]), ' places would lose CKZ money if FN ends')
    print('their ESI is ', simulatedcdfyear.loc[myFNeffect>0,'economic-strength-index-prev3yearavg'].mean())
    print(len(simulatedcdfyear.loc[myFNeffect<0]), ' places would gain CKZ money if FN ends')
    print('their ESI is ', simulatedcdfyear.loc[myFNeffect<0,'economic-strength-index-prev3yearavg'].mean())
    print(len(simulatedcdfyear.loc[myFNeffect==0]), ' places + tokyo would see no difference')
    print('their ESI is ',  simulatedcdfyear.loc[myFNeffect==0,'economic-strength-index-prev3yearavg'].mean())

    totalFNeffect = (simulatedcdfyear['netgainminusdeductions']+simulatedcdfyear['ckz']-simulatedcdfyear['ckz-noFN'])

    print(len(simulatedcdfyear.loc[totalFNeffect>0]), ' places would lose total money if FN ends')
    print('their ESI is ', simulatedcdfyear.loc[totalFNeffect>0,'economic-strength-index-prev3yearavg'].mean())
    print(len(simulatedcdfyear.loc[totalFNeffect<0]), ' places would gain total money if FN ends')
    print('their ESI is ', simulatedcdfyear.loc[totalFNeffect<0,'economic-strength-index-prev3yearavg'].mean())
    print(len(simulatedcdfyear.loc[totalFNeffect==0]), ' places would see no difference')
    print('their ESI is ',  simulatedcdfyear.loc[totalFNeffect==0,'economic-strength-index-prev3yearavg'].mean())

    simulatedcdfyear.loc[simulatedcdfyear['netgainminusdeductions']<0]
    
    output_df_year = simulatedcdfyear[['prefecture', 'code', 'ckz', 'ckz-noFN','economic-strength-index-prev3yearavg']].copy()
    ## re-add tokyo 23 this is easy since they just get no ckz
    tokyo23codes = [str(13101 + i) for i in range(23)]
    tokyo23df = pd.DataFrame(tokyo23codes,columns=['code'])
    tokyo23df['ckz'] = 0
    tokyo23df['ckz-noFN'] = 0
    tokyo23df['prefecture'] = '東京都'
    tokyo23df['economic-strength-index-prev3yearavg'] = np.nan
    output_df_year = pd.concat([output_df_year,tokyo23df])
    output_df_year['year'] = year

    output_df = pd.concat([output_df,output_df_year])

ckz_sim_df = output_df 

### Here I want to do some checks of ckz time trends to see whether its a good assumption to assume ckz in R5 is like R4,
### and how minimal can I make the assumptions
###

## First some global checks
fig, ax = cb.handlers()
ax.plot(cdf.year.unique(),cdf.groupby('year').sum()['ckz'],label='ckz')
#ax.plot(cdf.year.unique(),cdf.groupby('year').sum()['income'],label='income')
#ax.plot(cdf.year.unique(),cdf.groupby('year').sum()['final-demand'],label='final-demand')
#ax.plot(cdf.year.unique(),cdf.groupby('year').sum()['demand-pre-debt'],label='demand-pre-debt')
ax.legend()
fig.savefig('./simplots/muni-ckz-over-time.pdf')
plt.close('all')

## Now pick a random city and do some checks
code = np.random.choice(cdf['code'].unique())
citydf = cdf.loc[cdf['code']==code].sort_values('year')
print(citydf['city'].values[0],citydf['prefecture'].values[0])
fig, ax = cb.handlers()
ax.plot(citydf.year,citydf['ckz'],label='ckz')
ax.plot(citydf.year,citydf['income'],label='income')
ax.plot(citydf.year,citydf['final-demand'],label='final-demand')
ax.plot(citydf.year,citydf['demand-pre-debt'],label='demand-pre-debt')
ax.legend()
fig.savefig('./simplots/random-city-ckz-over-time.pdf')
plt.close('all')
########################
###### PREF SIM ########
########################

output_pref_df = pd.DataFrame()
simStartYear = np.min(fdf.year)
simEndYear =  np.max(cdf_pref.year)-1
for year in range(simStartYear, simEndYear+1):
    chihoukoufuzeiyear = year+1 #TODO: ACTUALLY THIS SHOULD BE +2. Donations are made in year X, decline in income is reported in year X+1, and then compensation occurs in X+2年度. Implement an estimation of the effect by using the last year's average or something...
    print(year)
    fdfyear = fdf.loc[fdf.year == year]
    ## keep only the prefectures 
    fdfyear = fdfyear.drop(fdfyear.loc[fdfyear['city']!='prefecture'].index)
    cdfyear = cdf_pref.loc[cdf_pref.year == chihoukoufuzeiyear]
    assert(len(cdfyear) == len(fdfyear))

    simulatedcdfyear = cdfyear.copy()
    simulatedcdfyear = simulatedcdfyear.merge(fdfyear,on=['prefecture'],validate='one_to_one', suffixes=['','_fdf'])
    simulatedcdfyear['ckz-preAdj'] = simulatedcdfyear['final-demand']-simulatedcdfyear['income'] ## Amount of ckz before final adjustment
    simulatedcdfyear.loc[simulatedcdfyear['ckz-preAdj']<0,'ckz-preAdj'] = 0 ## ckz cant be negative

    simulatedcdfyear.loc[simulatedcdfyear['economic-strength-index-prev3yearavg'].isna(),'economic-strength-index-prev3yearavg'] = simulatedcdfyear['economic-strength-index'] ## For places without a 3-year ESI, use current year ESI

    ## The debt scaling constant that was used for each pref in practice
    simulatedcdfyear['debt-scaling-factor'] = simulatedcdfyear['special-debt']/(simulatedcdfyear['demand-pre-debt']-simulatedcdfyear['income'])/simulatedcdfyear['economic-strength-index-prev3yearavg']
    # The adjustment factor that was used to reduce the final demand and get the ckz
    simulatedcdfyear['adjustment-factor'] =  1-(simulatedcdfyear['ckz']+simulatedcdfyear['income'])/simulatedcdfyear['final-demand']
    simulatedcdfyear.loc[simulatedcdfyear['ckz']==0,'adjustment-factor'] = 0

    ## Now to undo the FN effect add back 0.75*deductions
    simulatedcdfyear['income-noFN'] = simulatedcdfyear['income']+0.75*simulatedcdfyear['deductions']
    ## With this higher income, recompute how much debt they would have been allowed to issue

    if shortfall[chihoukoufuzeiyear]>0:
        subsidyFraction = 0.5
    else:
        subsidyFraction = 0
    print('Using subsidy fraction', subsidyFraction)
    governmentSubsidyTotal = subsidyFraction* 0.75*simulatedcdfyear['deductions'].sum() ## For testing purposes

    ## Find a uniform scaling of the debt-scaling-factors such that with the nouzei undone we still get the right total ckz-preAdj
    from scipy.optimize import minimize
    def ckzPreAdjConstraint(extraFactor):
        simulatedcdfyear['special-debt-noFN'] =  (extraFactor*simulatedcdfyear['debt-scaling-factor']*(simulatedcdfyear['demand-pre-debt']-simulatedcdfyear['income-noFN'])*simulatedcdfyear['economic-strength-index-prev3yearavg'])
        simulatedcdfyear['final-demand-noFN'] = simulatedcdfyear['demand-pre-debt']-simulatedcdfyear['special-debt-noFN']
        simulatedcdfyear['ckz-preAdj-noFN'] = simulatedcdfyear['final-demand-noFN']-simulatedcdfyear['income-noFN'] 
        simulatedcdfyear.loc[simulatedcdfyear['ckz-preAdj-noFN']<0,'ckz-preAdj-noFN'] = 0
        return np.abs(simulatedcdfyear['ckz-preAdj-noFN'].sum() - (simulatedcdfyear['ckz-preAdj'].sum() - governmentSubsidyTotal))

    res = minimize(ckzPreAdjConstraint, 1)
    extraDebtFactorFound = res.x
    simulatedcdfyear['debt-scaling-factor-noFN'] = extraDebtFactorFound*simulatedcdfyear['debt-scaling-factor']
    simulatedcdfyear['special-debt-noFN'] =  (simulatedcdfyear['debt-scaling-factor-noFN']*(simulatedcdfyear['demand-pre-debt']-simulatedcdfyear['income-noFN'])*simulatedcdfyear['economic-strength-index-prev3yearavg'])
    simulatedcdfyear['final-demand-noFN'] = simulatedcdfyear['demand-pre-debt']-simulatedcdfyear['special-debt-noFN']
    simulatedcdfyear['ckz-preAdj-noFN'] = simulatedcdfyear['final-demand-noFN']-simulatedcdfyear['income-noFN'] 
    simulatedcdfyear.loc[simulatedcdfyear['ckz-preAdj-noFN']<0,'ckz-preAdj-noFN'] = 0
    #assert(np.abs(1-simulatedcdfyear['ckz-preAdj-noFN'].sum()/simulatedcdfyear['ckz-preAdj'].sum())<0.001)

    ## Now do the same for the adjustment factor.
    ## Find a uniform scaling of the adjustment-factors such that with the nouzei undone we still get the right total ckz
    def ckzConstraint(extraFactor):
        simulatedcdfyear['ckz-noFN'] = simulatedcdfyear['final-demand-noFN'] * (1- extraFactor * simulatedcdfyear['adjustment-factor']) - simulatedcdfyear['income-noFN']
        simulatedcdfyear.loc[simulatedcdfyear['ckz-noFN']<0,'ckz-noFN'] = 0
        return np.abs(simulatedcdfyear['ckz-noFN'].sum()-(simulatedcdfyear['ckz'].sum()-governmentSubsidyTotal)) 

    res = minimize(ckzConstraint, 1)
    extraAdjFactorFound = res.x
    simulatedcdfyear['adjustment-factor-noFN'] = extraAdjFactorFound*simulatedcdfyear['adjustment-factor']
    simulatedcdfyear['ckz-noFN'] = simulatedcdfyear['final-demand-noFN'] * (1- simulatedcdfyear['adjustment-factor-noFN']) - simulatedcdfyear['income-noFN']
    simulatedcdfyear.loc[simulatedcdfyear['ckz-noFN']<0,'ckz-noFN'] = 0
    #assert(np.abs(1-simulatedcdfyear['ckz-noFN'].sum()/simulatedcdfyear['ckz'].sum())<0.001)
    
    simulatedcdfyear['ckz-noFN'].sum()/simulatedcdfyear['ckz'].sum()

    # In principle if the money is subsidized most places should get more ckz with FN than without, since FN decreases their income.
    myFNeffect = (simulatedcdfyear['ckz']-simulatedcdfyear['ckz-noFN'])

    print(len(simulatedcdfyear.loc[myFNeffect>0]), ' places would lose CKZ money if FN ends')
    print('their ESI is ', simulatedcdfyear.loc[myFNeffect>0,'economic-strength-index-prev3yearavg'].mean())
    print(len(simulatedcdfyear.loc[myFNeffect<0]), ' places would gain CKZ money if FN ends')
    print('their ESI is ', simulatedcdfyear.loc[myFNeffect<0,'economic-strength-index-prev3yearavg'].mean())
    print(len(simulatedcdfyear.loc[myFNeffect==0]), ' places + tokyo would see no difference')
    print('their ESI is ',  simulatedcdfyear.loc[myFNeffect==0,'economic-strength-index-prev3yearavg'].mean())

    totalFNeffect = (simulatedcdfyear['netgainminusdeductions']+simulatedcdfyear['ckz']-simulatedcdfyear['ckz-noFN'])

    print(len(simulatedcdfyear.loc[totalFNeffect>0]), ' places would lose total money if FN ends')
    print('their ESI is ', simulatedcdfyear.loc[totalFNeffect>0,'economic-strength-index-prev3yearavg'].mean())
    print(len(simulatedcdfyear.loc[totalFNeffect<0]), ' places would gain total money if FN ends')
    print('their ESI is ', simulatedcdfyear.loc[totalFNeffect<0,'economic-strength-index-prev3yearavg'].mean())
    print(len(simulatedcdfyear.loc[totalFNeffect==0]), ' places would see no difference')
    print('their ESI is ',  simulatedcdfyear.loc[totalFNeffect==0,'economic-strength-index-prev3yearavg'].mean())
    
    output_df_year = simulatedcdfyear[['prefecture', 'ckz', 'ckz-noFN','economic-strength-index-prev3yearavg']].copy()
    output_df_year['year'] = year

    output_pref_df = pd.concat([output_pref_df,output_df_year])

ckz_sim_pref_df = output_pref_df 

## Next steps 
## Look at whether there is a rollover effect from the debt in the following year.
## Look at how the subsidy money is distributed -- actually put into the same futsuu money pool? 


