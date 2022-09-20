import pandas as pd
import geopandas as gpd
import numpy as np

from japandata.furusatonouzei.data import furusato_df, furusato_rough_df

## furusato_df contains all the data including the donations directly to prefectures.

## This two contain just donations data, but cover the earliest years of the program
rough_annual_df = furusato_rough_df.groupby('year').sum().reset_index()

#FN at the prefectural level
furusato_pref_df = furusato_df.groupby(['prefecture','year']).sum().reset_index().drop(['flag'],axis=1)

#FN at the annual level
annual_df = furusato_pref_df.groupby(['year']).sum().reset_index()

#FN aggregated across all years
furusato_sumoveryears_df = furusato_df.groupby(['code','prefecturecity','prefecture', 'city']).sum().reset_index().drop('year', axis=1)

### Now we build a dataframe which contains a lot of extra useful data for the municipalities

###################################
##### Adding population data ######
###################################
from japandata.population.data import local_pop_df, prefecture_pop_df

fn_pop_df = pd.merge(furusato_df, local_pop_df, on=["code", "year", "prefecture"],validate='one_to_one', suffixes=['','_pop'])
fn_pop_df = fn_pop_df.drop('city_pop', axis=1)
assert(len(furusato_df.loc[~furusato_df['code'].isin(fn_pop_df['code']) & ~(furusato_df['city']=='prefecture')]) == 0)

fn_pop_pref_df = pd.merge(furusato_pref_df, prefecture_pop_df, on=["year","prefecture"],validate='one_to_one')
assert(len(fn_pop_pref_df) == len(furusato_pref_df))

fn_rough_pop_df = pd.merge(furusato_rough_df, local_pop_df, on=["code", "year", "prefecture"],validate='one_to_one', suffixes=['','_pop'])
fn_rough_pop_df = fn_rough_pop_df.drop('city_pop', axis=1)
assert(len(furusato_rough_df.loc[~furusato_rough_df['code'].isin(fn_rough_pop_df['code']) & ~(furusato_rough_df['city']=='prefecture')]) == 0)
###################################
##### Adding economic data ######
###################################
from japandata.indices.data import local_ind_df, pref_ind_df, prefmean_ind_df

## Clone 2020 data for 2021 
year_clone_df = local_ind_df.loc[local_ind_df['year'] == 2020].copy()
year_clone_df['year'] = 2021
local_ind_df = pd.concat([local_ind_df, year_clone_df])
year_clone_df = prefmean_ind_df.loc[prefmean_ind_df['year'] == 2020].copy()
year_clone_df['year'] = 2021
prefmean_ind_df = pd.concat([prefmean_ind_df, year_clone_df])

fn_pop_ind_df = pd.merge(fn_pop_df, local_ind_df, on=["code", "year", "prefecture"],validate='one_to_one', suffixes=['','_ind'])
assert(len(fn_pop_ind_df) == len(fn_pop_df))

fn_pop_ind_pref_df = pd.merge(fn_pop_pref_df, prefmean_ind_df, on=["year","prefecture"],validate='one_to_one')

fn_rough_pop_ind_df = pd.merge(fn_rough_pop_df, local_ind_df, on=["code", "year", "prefecture"],validate='one_to_one', suffixes=['','_ind'])
assert(len(fn_rough_pop_ind_df) == len(fn_rough_pop_df))
##################################
##### Adding readings data ######
#################################
from japandata.readings.data import names_df,pref_names_df

fn_pop_ind_reading_df = pd.merge(fn_pop_ind_df, names_df[['code','prefecture-reading','city-reading']],how='left', on="code", validate="many_to_one")
assert(len(fn_pop_ind_reading_df)==len(fn_pop_ind_df))

fn_pop_ind_pref_reading_df = pd.merge(fn_pop_ind_pref_df, pref_names_df[['code','prefecture-reading']],how='left', on="code", validate="many_to_one")
assert(len(fn_pop_ind_pref_reading_df)==len(fn_pop_ind_pref_df))

fn_rough_pop_ind_reading_df = pd.merge(fn_rough_pop_ind_df, names_df[['code','prefecture-reading','city-reading']],how='left', on="code", validate="many_to_one")
assert(len(fn_rough_pop_ind_reading_df)==len(fn_rough_pop_ind_df))
######################################
### Aliasing ####
######################################

local_df = fn_pop_ind_reading_df
pref_df = fn_pop_ind_pref_reading_df
rough_df = fn_rough_pop_ind_reading_df

#########################################################
###### Computing some per-person and fraction columns ###
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
rough_df_no23 = rough_df.drop(rough_df.loc[rough_df['code'].isin(tokyo23codes)].index)

# ######################################################
# ### Adding map  ###
########################################################
from japandata.maps.data import load_map

def local_map_df_loader(quality='coarse'):
    map_df = pd.DataFrame()
    for year in fn_pop_ind_df['year'].unique():
        map_df_year = load_map(year,level='local_dc',quality=quality)
        
        map_df_year = map_df_year.drop(['bureau', 'county', 'special'],axis=1)
        map_df_year['year'] = year
        map_df = pd.concat([map_df,map_df_year])

    map_df = map_df.drop(map_df.loc[~map_df['code'].isin(fn_pop_ind_df['code'])].index,axis=0)
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
    for year in fn_pop_ind_pref_df['year'].unique():
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

