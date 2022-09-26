import os
import pandas as pd

CACHE_FOLDER = os.path.join(os.path.dirname(__file__),'cache/')
furusato_df_CACHE = os.path.join(CACHE_FOLDER, 'furusato_df.parquet')
furusato_rough_df_CACHE = os.path.join(CACHE_FOLDER, 'furusato_rough_df.parquet')
pref_df_CACHE = os.path.join(CACHE_FOLDER, 'pref_df.parquet')

try:
    furusato_df = pd.read_parquet(furusato_df_CACHE)
    furusato_rough_df = pd.read_parquet(furusato_rough_df_CACHE)
    pref_df = pd.read_parquet(pref_df_CACHE)
except FileNotFoundError:
    import geopandas as gpd
    import numpy as np

    from japandata.furusatonouzei.data import furusato_df, furusato_rough_df

    ## furusato_df contains all the data including the donations directly to prefectures.
    ## furusato_rough_df contains just donations data, but cover the earliest years of the program

    ########################
    ### Adding ckz data ####
    ########################

    from sim import ckz_sim_df

    assert((len(ckz_sim_df) == len(furusato_df)))
    furusato_ckz_df = pd.merge(furusato_df,ckz_sim_df, on=["prefecture", "code", "year"],validate='one_to_one', suffixes=['','_ckz'])
    assert((len(furusato_ckz_df) == len(furusato_df)))
    furusato_df = furusato_ckz_df

    ## TODO ADD ROUGH HERE

    ##################################
    ##### Adding readings data ######
    #################################
    from japandata.readings.data import names_df,pref_names_df

    furusato_df['prefecture-reading'] = pref_names_df.set_index('prefecture').loc[furusato_df['prefecture'],'prefecture-reading'].values
    furusato_df['city-reading'] = np.nan
    furusato_df.loc[furusato_df['code'].isin(names_df.code),'city-reading'] = names_df.set_index('code').loc[furusato_df.loc[furusato_df['code'].isin(names_df.code),'code'],'city-reading'].values

    ### TODO ADD ROUGH HERE
    #fn_rough_pop_ind_reading_df = pd.merge(fn_rough_pop_ind_df, names_df[['code','prefecture-reading','city-reading']],how='left', on="code", validate="many_to_one")
    #assert(len(fn_rough_pop_ind_reading_df)==len(fn_rough_pop_ind_df))

    ###########################################################
    ############# Summing to get the prefecture total #########
    ###########################################################

    pref_df = furusato_df.groupby(['prefecture','year','prefecture-reading']).sum(numeric_only=True).reset_index().drop(['flag'],axis=1)
    assert(furusato_df.loc[furusato_df['city']=='prefecture', 'deductions'].sum()==pref_df['pref-tax-deductions'].sum())

    #TODO ADD ROUGH HERE 

    ###################################
    ##### Adding population data ######
    ###################################

    from japandata.population.data import local_pop_df, prefecture_pop_df

    prefecture_pop_df['code'] = prefecture_pop_df['code']+'000'
    local_pop_df = local_pop_df.drop(['city','code6digit'],axis=1)
    assert(set(local_pop_df.columns) == set(prefecture_pop_df.columns) )
    pop_df = pd.concat([local_pop_df,prefecture_pop_df])

    furusato_pop_df = pd.merge(furusato_df, pop_df, on=["code", "year"],validate='one_to_one', suffixes=['','_pop'])
    assert(len(furusato_pop_df) == len(furusato_df))
    furusato_df = furusato_pop_df

    pref_pop_df = pd.merge(pref_df, prefecture_pop_df, on=["year","prefecture"],validate='one_to_one')
    assert(len(pref_pop_df) == len(pref_df))
    pref_df = pref_pop_df

    #TODO ADD ROUGH HERE 
    # fn_rough_pop_df = pd.merge(furusato_rough_df, local_pop_df, on=["code", "year", "prefecture"],validate='one_to_one', suffixes=['','_pop'])
    # fn_rough_pop_df = fn_rough_pop_df.drop('city_pop', axis=1)
    # assert(len(furusato_rough_df.loc[~furusato_rough_df['code'].isin(fn_rough_pop_df['code']) & ~(furusato_rough_df['city']=='prefecture')]) == 0)

    ###################################
    ##### Adding economic data ########
    ################################### 
    from japandata.indices.data import local_ind_df, pref_ind_df, prefmean_ind_df

    ## Clone 2020 data for 2021 ##TODO AUTOMATE THIS
    year_clone_df = local_ind_df.loc[local_ind_df['year'] == 2020].copy()
    year_clone_df['year'] = 2021
    local_ind_df = pd.concat([local_ind_df, year_clone_df])
    year_clone_df = prefmean_ind_df.loc[prefmean_ind_df['year'] == 2020].copy()
    year_clone_df['year'] = 2021
    prefmean_ind_df = pd.concat([prefmean_ind_df, year_clone_df])
    local_ind_df = local_ind_df.drop(['city'], axis=1)
    prefmean_ind_df['code'] = furusato_df.loc[(furusato_df['city']=='prefecture') & (furusato_df['year'] == 2020),['prefecture','code']].set_index('prefecture').loc[prefmean_ind_df['prefecture']].values
    ind_df = pd.concat([local_ind_df, prefmean_ind_df])

    furusato_ind_df = pd.merge(furusato_df, ind_df, on=["code", "year", "prefecture"],validate='one_to_one', suffixes=['','_ind'])
    assert(len(furusato_ind_df) == len(furusato_df))
    furusato_df = furusato_ind_df

    pref_ind_df = pd.merge(pref_df, prefmean_ind_df, on=["year","prefecture"],validate='one_to_one')
    assert(len(pref_ind_df) == len(pref_df))
    pref_df = pref_ind_df

    # TODO ADD ROUGH HERE

    # fn_rough_pop_ind_df = pd.merge(fn_rough_pop_df, local_ind_df, on=["code", "year", "prefecture"],validate='one_to_one', suffixes=['','_ind'])
    # assert(len(fn_rough_pop_ind_df) == len(fn_rough_pop_df))


    #########################################################
    ###### Computing some per-person and fraction columns ###
    #########################################################

    pd.options.mode.chained_assignment = None

    furusato_df['profit-from-ckz'] = furusato_df['ckz']-furusato_df['ckz-noFN']
    furusato_df['profit-incl-ckz'] = furusato_df['netgainminusdeductions']+furusato_df['profit-from-ckz']
    furusato_df['profit-per-person'] = furusato_df['netgainminusdeductions']/furusato_df['total-pop']
    furusato_df['profit-per-person-incl-ckz'] = furusato_df['profit-incl-ckz']/furusato_df['total-pop']

    pref_df['profit-from-ckz'] = pref_df['ckz']-pref_df['ckz-noFN']
    pref_df['profit-incl-ckz'] = pref_df['netgainminusdeductions']+pref_df['profit-from-ckz']
    pref_df['profit-per-person'] = pref_df['netgainminusdeductions']/pref_df['total-pop']
    pref_df['profit-per-person-incl-ckz'] = pref_df['profit-incl-ckz']/pref_df['total-pop']


    def prefProfitShare(row, includeCkzEffect=False):
        print(row.name)
        if row['city']=='prefecture':
            return np.nan
        else:
            correspondingPrefRow = furusato_df.loc[(furusato_df['prefecture']==row['prefecture']) & (furusato_df['year']==row['year']) & (furusato_df['city']=='prefecture')]
            share = (correspondingPrefRow['donations']-correspondingPrefRow['deductions'] + int(includeCkzEffect) *correspondingPrefRow['profit-from-ckz'])*row['total-pop']/correspondingPrefRow['total-pop']
            assert(share.shape == (1,))
            #print(1-(share.values[0]/(row['pref-tax-deductions'])))
            return share.values[0]

    furusato_df['share-of-pref-profit'] = furusato_df.apply(prefProfitShare,axis=1)
    furusato_df['share-of-pref-profit-incl-ckz'] = furusato_df.apply(prefProfitShare,axis=1,includeCkzEffect=True)

    assert(furusato_df.loc[ (furusato_df['city']=='prefecture'), 'donations'].sum() - furusato_df['pref-tax-deductions'].sum() == furusato_df['share-of-pref-profit'].sum())

    furusato_df['profit-incl-pref-share'] = furusato_df['netgainminusdeductions']+furusato_df['share-of-pref-profit']
    furusato_df['profit-incl-pref-share-incl-ckz'] = furusato_df['profit-incl-ckz']+furusato_df['share-of-pref-profit-incl-ckz']

    furusato_df['profit-per-person-incl-pref-share'] = furusato_df['profit-incl-pref-share']/furusato_df['total-pop']
    furusato_df['profit-per-person-incl-pref-share-incl-ckz'] = furusato_df['profit-incl-pref-share-incl-ckz']/furusato_df['total-pop']


    # totalbyyear = local_df.groupby('year').sum()['donations']
    # local_df['donations-fraction'] = local_df.apply(lambda row: row['donations']/totalbyyear[row['year']],axis=1)
    # local_df['donations-per-person'] = local_df.apply(lambda row: row['donations']/row['total-pop'],axis=1)

    # pref_df['donations-fraction'] = pref_df.apply(lambda row: row['donations']/totalbyyear[row['year']],axis=1)
    # pref_df['donations-per-person'] = pref_df.apply(lambda row: row['donations']/row['total-pop'],axis=1)

    ########## CACHING ##########
    furusato_df.to_parquet(furusato_df_CACHE)
    furusato_rough_df.to_parquet(furusato_rough_df_CACHE)
    pref_df.to_parquet(pref_df_CACHE)


####################################
############# Aliasing ############
####################################

local_df =  furusato_df.loc[furusato_df['city']!='prefecture']
#just_pref_df  = furusato_df.loc[furusato_df['city']=='prefecture']

annual_df = pref_df.groupby(['year']).sum().reset_index()
rough_annual_df = furusato_rough_df.groupby('year').sum().reset_index()


tokyo23codes = [str(13101 + i) for i in range(23)]
local_df_no23 = local_df.drop(local_df.loc[local_df['code'].isin(tokyo23codes)].index)
#rough_df_no23 = rough_df.drop(rough_df.loc[rough_df['code'].isin(tokyo23codes)].index)

# ###################
# ### Adding map  ###
#####################
from japandata.maps.data import load_map

def local_map_df_loader(quality='coarse'):
    map_df = pd.DataFrame()
    for year in local_df['year'].unique():
        map_df_year = load_map(year,level='local_dc',quality=quality)
        
        map_df_year = map_df_year.drop(['bureau', 'county', 'special'],axis=1)
        map_df_year['year'] = year
        map_df = pd.concat([map_df,map_df_year])

    map_df = map_df.drop(map_df.loc[~map_df['code'].isin(local_df['code'])].index,axis=0)
    map_df=map_df.reset_index(drop=True)

    local_map_df = pd.merge(local_df, map_df, on=["year","prefecture", "code"],how='left', suffixes=['','_map'])
    local_map_df= local_map_df.drop('city_map', axis=1)
    for index, row in local_map_df.iterrows():
        if row.geometry==None:
            local_map_df.at[index, 'geometry'] = map_df.loc[(map_df['code']==row.code) & (map_df['prefecture']==row.prefecture),'geometry'].values[0]
        else:
            pass
    assert(len(local_map_df) == len(local_df))
    local_map_df = gpd.GeoDataFrame(local_map_df)
    return local_map_df

def pref_map_df_loader(quality='stylized'):
    map_df = pd.DataFrame()
    for year in pref_df['year'].unique():
        map_df_year = load_map(year,level='prefecture', quality=quality)
        map_df_year['year'] = year
        map_df = pd.concat([map_df,map_df_year])
    map_df=map_df.reset_index(drop=True)

    pref_map_df = pd.merge(pref_df, map_df, on=["year","prefecture"],validate='one_to_one', suffixes=['','_map'])
    pref_map_df = gpd.GeoDataFrame(pref_map_df)
    return pref_map_df

# # ##################################
# # ### To save to geojson  ##########
# # ##################################

# gpd.GeoDataFrame(local_df).to_file('./local.geojson',driver='GeoJSON')
# gpd.GeoDataFrame(pref_df).to_file('./pref.geojson',driver='GeoJSON')

