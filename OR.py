# import json
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from dateutil.parser import *
import ConfigParser
import sys, getopt
import os

def operator_ranking_12m(dataAll, curr_end_date, Min_Well):
    data_sort_API_plot = dataAll

    #data_sort_API_plot['First_Prod_Year'] = data_sort_API_plot['First Prod Date'].apply(lambda x: int(x[:4]))
    data_sort_API_plot['Date'] = data_sort_API_plot['First Prod Date'].apply(lambda x: parse(x).date())

    end_date = curr_end_date - relativedelta(months = 12)
    start_date = end_date - relativedelta(months = 12)
    #

    data_sort_API_new0 = data_sort_API_plot[data_sort_API_plot['Date'] < end_date]
    data_sort_API_new = data_sort_API_new0[data_sort_API_new0['Date'] >= start_date]

    for idx in data_sort_API_new['State'].index:
        if data_sort_API_new['State'].loc[idx] == 'OHIO':
            if data_sort_API_new['Date'].loc[idx] >= dt.date(2017,4,1) - relativedelta(months = 12):
                data_sort_API_new.drop([idx], inplace = True)
                # continue
        elif data_sort_API_new['State'].loc[idx] == 'WEST VIRGINIA':
            if data_sort_API_new['Date'].loc[idx] >= dt.date(2017, 1, 1) - relativedelta(months=12):
                data_sort_API_new.drop([idx], inplace=True)
        else:
            pass
                # continue
    # data_sort_API_new = data_sort_API_plot

    data_sort_API_new['WELL_CNT'] = data_sort_API_new['First Prod Date'].apply(lambda x: 1)

    # data_sort_API_new = data_sort_API_new[data_sort_API_new['LAT_LENGTH'] != 0]

    data_sort_API_OPER = data_sort_API_new.groupby('Operator Name').mean()
    data_sort_API_OPER_sum = data_sort_API_new.groupby('Operator Name').sum()
    data_sort_API_OPER['WELL_CNT'] = data_sort_API_OPER_sum['WELL_CNT']

    data_sort_API_exl_lat0 = data_sort_API_new[data_sort_API_new['LAT_LENGTH'] != 0]
    data_sort_API_OPER_exl_lat0 = data_sort_API_exl_lat0.groupby('Operator Name').mean()
    data_sort_API_OPER['LAT_LENGTH'] = data_sort_API_OPER_exl_lat0['LAT_LENGTH']
    data_sort_API_OPER['FIRST12REV_PER_FT'] = data_sort_API_OPER['FIRST12_REV']/data_sort_API_OPER['LAT_LENGTH']*1000000
    data_sort_API_OPER['FIRST6REV_PER_FT'] = data_sort_API_OPER['FIRST6_REV'] / data_sort_API_OPER['LAT_LENGTH']*1000000
    data_sort_API_OPER['FIRST3REV_PER_FT'] = data_sort_API_OPER['FIRST3_REV'] / data_sort_API_OPER['LAT_LENGTH']*1000000

    data_sort_API_OPER_5Plus_12 = data_sort_API_OPER[data_sort_API_OPER['WELL_CNT'] >= Min_Well]
    data_sort_API_OPER_5Plus_6 = data_sort_API_OPER[data_sort_API_OPER['WELL_CNT'] >= Min_Well]
    data_sort_API_OPER_5Plus_3 = data_sort_API_OPER[data_sort_API_OPER['WELL_CNT'] >= Min_Well]

    data_sort_API_OPER.sort_values(by = 'WELL_CNT', axis = 0, ascending = False, inplace = True)
    data_sort_API_OPER_5Plus_12.sort_values(by='FIRST12_REV', axis=0, ascending=False, inplace=True)
    data_sort_API_OPER_5Plus_6.sort_values(by='FIRST6_REV', axis=0, ascending=False, inplace=True)
    data_sort_API_OPER_5Plus_3.sort_values(by='FIRST3_REV', axis=0, ascending=False, inplace=True)

    data_sort_API_OPER_5Plus_12['Rank'] = np.arange(1, data_sort_API_OPER_5Plus_12.shape[0] + 1)
    data_sort_API_OPER_5Plus_12 = data_sort_API_OPER_5Plus_12.reset_index(drop=False)
    data_sort_API_OPER_5Plus_12.set_index('Rank', inplace=True)

    data_sort_API_OPER_5Plus_6['Rank'] = np.arange(1, data_sort_API_OPER_5Plus_6.shape[0] + 1)
    data_sort_API_OPER_5Plus_6 = data_sort_API_OPER_5Plus_6.reset_index(drop=False)
    data_sort_API_OPER_5Plus_6.set_index('Rank', inplace=True)

    data_sort_API_OPER_5Plus_3['Rank'] = np.arange(1, data_sort_API_OPER_5Plus_3.shape[0] + 1)
    data_sort_API_OPER_5Plus_3 = data_sort_API_OPER_5Plus_3.reset_index(drop=False)
    data_sort_API_OPER_5Plus_3.set_index('Rank', inplace=True)

    # Just for exporting to Excel purpose, May means Jun 1
    end_date = end_date - relativedelta(months = 1)

    return (data_sort_API_OPER, data_sort_API_OPER_5Plus_12, data_sort_API_OPER_5Plus_6, data_sort_API_OPER_5Plus_3, start_date, end_date)

    # data_sort_API_OPER[['FIRST12_REV','FIRST12_BOE', 'FIRST12_LIQ', 'FIRST12_GAS', 'LAT_LENGTH','WELL_CNT']].to_csv('latest prod.csv')

def operator_ranking_6m(dataAll, curr_end_date, Min_Well):
    data_sort_API_plot = dataAll

    #data_sort_API_plot['First_Prod_Year'] = data_sort_API_plot['First Prod Date'].apply(lambda x: int(x[:4]))
    data_sort_API_plot['Date'] = data_sort_API_plot['First Prod Date'].apply(lambda x: parse(x).date())

    end_date = curr_end_date - relativedelta(months = 6)
    start_date = end_date - relativedelta(months = 12)

    data_sort_API_new0 = data_sort_API_plot[data_sort_API_plot['Date'] < end_date]
    data_sort_API_new = data_sort_API_new0[data_sort_API_new0['Date'] >= start_date]
    # data_sort_API_new = data_sort_API_plot

    for idx in data_sort_API_new['State'].index:
        if data_sort_API_new['State'].loc[idx] == 'OHIO':
            if data_sort_API_new['Date'].loc[idx] >= dt.date(2017,4,1) - relativedelta(months = 6):
                data_sort_API_new.drop([idx], inplace = True)
                # continue
        elif data_sort_API_new['State'].loc[idx] == 'WEST VIRGINIA':
            if data_sort_API_new['Date'].loc[idx] >= dt.date(2017, 1, 1) - relativedelta(months=6):
                data_sort_API_new.drop([idx], inplace=True)
        else:
            pass

    data_sort_API_new['WELL_CNT'] = data_sort_API_new['First Prod Date'].apply(lambda x: 1)

    # data_sort_API_new = data_sort_API_new[data_sort_API_new['LAT_LENGTH'] != 0]

    data_sort_API_OPER = data_sort_API_new.groupby('Operator Name').mean()
    data_sort_API_OPER_sum = data_sort_API_new.groupby('Operator Name').sum()
    data_sort_API_OPER['WELL_CNT'] = data_sort_API_OPER_sum['WELL_CNT']

    data_sort_API_exl_lat0 = data_sort_API_new[data_sort_API_new['LAT_LENGTH'] != 0]
    data_sort_API_OPER_exl_lat0 = data_sort_API_exl_lat0.groupby('Operator Name').mean()
    data_sort_API_OPER['LAT_LENGTH'] = data_sort_API_OPER_exl_lat0['LAT_LENGTH']

    data_sort_API_OPER['FIRST6REV_PER_FT'] = data_sort_API_OPER['FIRST6_REV'] / data_sort_API_OPER['LAT_LENGTH']*1000000
    data_sort_API_OPER['FIRST3REV_PER_FT'] = data_sort_API_OPER['FIRST3_REV'] / data_sort_API_OPER['LAT_LENGTH']*1000000

    data_sort_API_OPER_5Plus_6 = data_sort_API_OPER[data_sort_API_OPER['WELL_CNT'] >= Min_Well]
    data_sort_API_OPER_5Plus_3 = data_sort_API_OPER[data_sort_API_OPER['WELL_CNT'] >= Min_Well]

    data_sort_API_OPER.sort_values(by='WELL_CNT', axis=0, ascending=False, inplace=True)
    data_sort_API_OPER_5Plus_6.sort_values(by='FIRST6_REV', axis=0, ascending=False, inplace=True)
    data_sort_API_OPER_5Plus_3.sort_values(by='FIRST3_REV', axis=0, ascending=False, inplace=True)

    data_sort_API_OPER_5Plus_6['Rank'] = np.arange(1, data_sort_API_OPER_5Plus_6.shape[0] + 1)
    data_sort_API_OPER_5Plus_6 = data_sort_API_OPER_5Plus_6.reset_index(drop=False)
    data_sort_API_OPER_5Plus_6.set_index('Rank', inplace=True)

    data_sort_API_OPER_5Plus_3['Rank'] = np.arange(1, data_sort_API_OPER_5Plus_3.shape[0] + 1)
    data_sort_API_OPER_5Plus_3 = data_sort_API_OPER_5Plus_3.reset_index(drop=False)
    data_sort_API_OPER_5Plus_3.set_index('Rank', inplace=True)
    # Just for exporting to Excel purpose, May means Jun 1
    end_date = end_date - relativedelta(months = 1)

    return (data_sort_API_OPER, data_sort_API_OPER_5Plus_6, data_sort_API_OPER_5Plus_3, start_date, end_date)

def operator_ranking_3m(dataAll, curr_end_date, Min_Well):
    data_sort_API_plot = dataAll

    #data_sort_API_plot['First_Prod_Year'] = data_sort_API_plot['First Prod Date'].apply(lambda x: int(x[:4]))
    data_sort_API_plot['Date'] = data_sort_API_plot['First Prod Date'].apply(lambda x: parse(x).date())

    end_date = curr_end_date - relativedelta(months = 3)
    start_date = end_date - relativedelta(months = 12)

    data_sort_API_new0 = data_sort_API_plot[data_sort_API_plot['Date'] < end_date]
    data_sort_API_new = data_sort_API_new0[data_sort_API_new0['Date'] >= start_date]

    for idx in data_sort_API_new['State'].index:
        if data_sort_API_new['State'].loc[idx] == 'OHIO':
            if data_sort_API_new['Date'].loc[idx] >= dt.date(2017,4,1) - relativedelta(months = 3):
                data_sort_API_new.drop([idx], inplace = True)
                # continue
        elif data_sort_API_new['State'].loc[idx] == 'WEST VIRGINIA':
            if data_sort_API_new['Date'].loc[idx] >= dt.date(2017, 1, 1) - relativedelta(months=3):
                data_sort_API_new.drop([idx], inplace=True)
        else:
            pass

    data_sort_API_new['WELL_CNT'] = data_sort_API_new['First Prod Date'].apply(lambda x: 1)

    # data_sort_API_new = data_sort_API_new[data_sort_API_new['LAT_LENGTH'] != 0]

    data_sort_API_OPER = data_sort_API_new.groupby('Operator Name').mean()
    data_sort_API_OPER_sum = data_sort_API_new.groupby('Operator Name').sum()
    data_sort_API_OPER['WELL_CNT'] = data_sort_API_OPER_sum['WELL_CNT']

    data_sort_API_exl_lat0 = data_sort_API_new[data_sort_API_new['LAT_LENGTH'] != 0]
    data_sort_API_OPER_exl_lat0 = data_sort_API_exl_lat0.groupby('Operator Name').mean()
    data_sort_API_OPER['LAT_LENGTH'] = data_sort_API_OPER_exl_lat0['LAT_LENGTH']

    data_sort_API_OPER['FIRST3REV_PER_FT'] = data_sort_API_OPER['FIRST3_REV'] / data_sort_API_OPER['LAT_LENGTH']*1000000

    data_sort_API_OPER_5Plus = data_sort_API_OPER[data_sort_API_OPER['WELL_CNT'] >= Min_Well]

    data_sort_API_OPER.sort_values(by='WELL_CNT', axis=0, ascending=False, inplace=True)
    data_sort_API_OPER_5Plus.sort_values(by='FIRST3_REV', axis=0, ascending=False, inplace=True)

    data_sort_API_OPER_5Plus['Rank'] = np.arange(1, data_sort_API_OPER_5Plus.shape[0] + 1)
    data_sort_API_OPER_5Plus = data_sort_API_OPER_5Plus.reset_index(drop=False)
    data_sort_API_OPER_5Plus.set_index('Rank', inplace=True)
    # Just for exporting to Excel purpose, May means Jun 1
    end_date = end_date - relativedelta(months = 1)

    return (data_sort_API_OPER, data_sort_API_OPER_5Plus, start_date, end_date)


def read_config(configFile):
    config = ConfigParser.ConfigParser()
    config.read(configFile)

    def configMap(section):
        config_map = {}
        entries = config.options(section)
        for entry in entries:
            try:
                config_map[entry] = config.get(section, entry)
                if config_map[entry] == -1:
                    DebugPrint("Skipped: %s" % entry)
                    # DebugPrint("Skipped: %s", % entry)
            except ValueError:
                raise ValueError("Exception: %s!" % entry)
                config_map[entry] = None
        return config_map

    Input_Directory = configMap("ORR")['input_directory']
    Input_IHS = configMap("ORR")['input_ihs']
    Output_Directory = configMap("ORR")['output_directory']
    Output_Name = configMap("ORR")['output_name']
    Oper_Name_Directory = configMap("ORR")['oper_name_directory']
    Oper_Name_File = configMap("ORR")['oper_name_file']
    Public_Comp_File = configMap("ORR")['public_comp_file']
    Oil_Price = float(configMap("ORR")['oil_price'])
    Gas_Price = float(configMap("ORR")['gas_price'])
    Curr_Date = parse(configMap("ORR")['curr_date']).date()
    # Oper_all = configMap("ORR")['operators']
    # Operators = Oper_all.split(',')
    Min_Well = configMap("ORR")['min_well']
    Min_Well = int(Min_Well)

    return (Input_Directory, Input_IHS, Output_Directory, Output_Name, Oper_Name_Directory, Oper_Name_File, Public_Comp_File, Oil_Price, Gas_Price, Curr_Date, Min_Well)


def main():

    try:
        configFile = sys.argv[1]
        (Input_Directory, Input_IHS, Output_Directory, Output_Name, Oper_Name_Directory, Oper_Name_File, Public_Comp_File, Oil_Price, Gas_Price, Curr_Date, Min_Well) = read_config(configFile)
    except:
        configFile = 'ORR_config.txt'
        (Input_Directory, Input_IHS, Output_Directory, Output_Name, Oper_Name_Directory, Oper_Name_File, Public_Comp_File, Oil_Price, Gas_Price, Curr_Date, Min_Well) = read_config(configFile)


    dataAll = pd.read_csv(os.path.join(Input_Directory, Input_IHS),
                          usecols=['Operator Name', 'Primary API', 'Latitude', 'Longitude',
                                   'First Prod Date','Hole Direction',
                                   'Gas Mos 2 thru 4', 'Gas Mos 2 thru 7', 'Gas Mos 2 thru 13', 'Oil Mos 2 thru 4',
                                   'Oil Mos 2 thru 7', 'Oil Mos 2 thru 13', 'Lower Perf', 'Upper Perf', 'State','TD','TVD'])

    if Output_Name == 'Haynesville':
        API_all = set(dataAll['Primary API'])
        datanew2 = dataAll.groupby('Primary API').sum()
        datadict = dataAll.set_index("Primary API", drop = True)
        datadict2 = datadict.to_dict()
        datanew2['Operator Name'] = 0
        datanew2['First Prod Date'] = 0
        datanew2['Hole Direction'] = 0
        datanew2['State'] = 0
        for API_one in API_all:
            datanew2.loc[API_one,'Operator Name'] = datadict2['Operator Name'][API_one]
            datanew2.loc[API_one, 'Latitude'] = datadict2['Latitude'][API_one]
            datanew2.loc[API_one, 'Longitude'] = datadict2['Longitude'][API_one]
            datanew2.loc[API_one, 'First Prod Date'] = datadict2['First Prod Date'][API_one]
            datanew2.loc[API_one, 'Hole Direction'] = datadict2['Hole Direction'][API_one]
            datanew2.loc[API_one, 'Lower Perf'] = datadict2['Lower Perf'][API_one]
            datanew2.loc[API_one, 'Upper Perf'] = datadict2['Upper Perf'][API_one]
            datanew2.loc[API_one, 'State'] = datadict2['State'][API_one]
        dataAll = datanew2.reset_index(inplace = False)

    curr_end_date = Curr_Date

    writer = pd.ExcelWriter(os.path.join(Output_Directory, 'ORR-' + Output_Name + ' as of ' + str(curr_end_date) +'.xlsx'))

    # dataAll = dataAll[dataAll['API_NO']!='0']
    dataAll.fillna(0, inplace=True)

    operator_name = pd.read_csv(os.path.join(Oper_Name_Directory, Oper_Name_File))

    for idx in operator_name.index:
        dataAll['Operator Name'] = dataAll['Operator Name'].apply(
            lambda x: operator_name['ToUse'].loc[idx] if x == operator_name['Original'].loc[idx] else x)

    public_comp_name = pd.read_csv(os.path.join(Oper_Name_Directory, Public_Comp_File))

    for idx in public_comp_name.index:
        dataAll['Operator Name'] = dataAll['Operator Name'].apply(
            lambda x: public_comp_name['Name'].loc[idx] if x == public_comp_name['Ticker'].loc[idx] else x)

    dataAll['FIRST12_LIQ'] = dataAll['Oil Mos 2 thru 13']
    dataAll['FIRST12_GAS'] = dataAll['Gas Mos 2 thru 13']
    dataAll['FIRST12_BOE'] = dataAll['FIRST12_LIQ'] + dataAll['FIRST12_GAS'] / 6.

    # dataAll['FIRST12_GOR'] = dataAll['FIRST12_GAS'] / dataAll['FIRST12_LIQ']
    dataAll['FIRST6_LIQ'] = dataAll['Oil Mos 2 thru 7']
    dataAll['FIRST6_GAS'] = dataAll['Gas Mos 2 thru 7']
    dataAll['FIRST6_BOE'] = dataAll['FIRST6_LIQ'] + dataAll['FIRST6_GAS'] / 6.

    dataAll['FIRST3_LIQ'] = dataAll['Oil Mos 2 thru 4']
    dataAll['FIRST3_GAS'] = dataAll['Gas Mos 2 thru 4']
    dataAll['FIRST3_BOE'] = dataAll['FIRST3_LIQ'] + dataAll['FIRST3_GAS'] / 6.

    dataAll['LAT_LENGTH'] = dataAll['Lower Perf'] - dataAll['Upper Perf']
    for idx2 in dataAll.index:
        if dataAll['LAT_LENGTH'].loc[idx2] == 0:
            dataAll['LAT_LENGTH'].loc[idx2] = dataAll['TD'].loc[idx2] - dataAll['TVD'].loc[idx2]
            if dataAll['TD'].loc[idx2] == 0 or dataAll['TVD'].loc[idx2] == 0:
                dataAll['LAT_LENGTH'].loc[idx2] = np.nan

    dataAll['LAT_LENGTH'] = dataAll['LAT_LENGTH'].apply(lambda x: 0 if x < 1500 else x)

    dataAll['FIRST12_REV'] = (dataAll['FIRST12_LIQ'] * Oil_Price + dataAll['FIRST12_GAS'] * Gas_Price) / 1000000.
    dataAll['FIRST6_REV'] = (dataAll['FIRST6_LIQ'] * Oil_Price + dataAll['FIRST6_GAS'] * Gas_Price) / 1000000.
    dataAll['FIRST3_REV'] = (dataAll['FIRST3_LIQ'] * Oil_Price + dataAll['FIRST3_GAS'] * Gas_Price) / 1000000.

    # dataAll[['Operator Name','First Prod Date', 'FIRST3_REV']].to_csv('123.csv')

    # dataAll['FIRST12_LIQREV'] = dataAll['FIRST12_LIQ'] * Oil_Price / 1000000.
    # dataAll['FIRST6_LIQREV'] = dataAll['FIRST6_LIQ'] * Oil_Price / 1000000.
    # dataAll['FIRST3_LIQREV'] = dataAll['FIRST3_LIQ'] * Oil_Price / 1000000.
    #
    # dataAll['FIRST12_GASREV'] = dataAll['FIRST12_GAS'] * Gas_Price / 1000000.
    # dataAll['FIRST6_GASREV'] = dataAll['FIRST6_GAS'] * Gas_Price / 1000000.
    # dataAll['FIRST3_GASREV'] = dataAll['FIRST3_GAS'] * Gas_Price / 1000000.

    #
    # 12 Month ORR
    # (data_sort_API_OPER_12, data_sort_API_OPER_5Plus_12_12, data_sort_API_OPER_5Plus_6_12, data_sort_API_OPER_5Plus_3_12, start_date_12, end_date_12) = operator_ranking_12m(dataAll, curr_end_date, Min_Well)
    # date_period_12 = str(start_date_12.strftime('%b'))+str(start_date_12.year)[2:4]+'-'+str(end_date_12.strftime('%b'))+str(end_date_12.year)[2:4]
    # 6 Month ORR
    # (data_sort_API_OPER_6, data_sort_API_OPER_5Plus_6_6, data_sort_API_OPER_5Plus_3_6, start_date_6, end_date_6) = operator_ranking_6m(dataAll, curr_end_date, Min_Well)
    # date_period_6 = str(start_date_6.strftime('%b')) + str(start_date_6.year)[2:4] + '-' + str(end_date_6.strftime('%b')) + str(end_date_6.year)[2:4]
    # 3 Month ORR
    (data_sort_API_OPER_3, data_sort_API_OPER_5Plus_3_3, start_date_3, end_date_3) = operator_ranking_3m(dataAll, curr_end_date, Min_Well)
    date_period_3 = str(start_date_3.strftime('%b')) + str(start_date_3.year)[2:4] + '-' + str(end_date_3.strftime('%b')) + str(end_date_3.year)[2:4]

    # data_sort_API_OPER_5Plus_12_12[['Operator Name','FIRST12_REV', 'FIRST12_BOE', 'FIRST12_LIQ', 'FIRST12_GAS', 'LAT_LENGTH', 'WELL_CNT',
    #                     'FIRST12REV_PER_FT']].to_excel(writer, 'First12M(' + str(Min_Well) + '+) btwn ' + date_period_12)
    # data_sort_API_OPER_5Plus_6_12[['Operator Name','FIRST6_REV', 'FIRST6_BOE', 'FIRST6_LIQ', 'FIRST6_GAS', 'LAT_LENGTH', 'WELL_CNT',
    #                        'FIRST6REV_PER_FT']].to_excel(writer, 'First6M(' + str(Min_Well) + '+) btwn ' + date_period_12)
    # data_sort_API_OPER_5Plus_3_12[['Operator Name','FIRST3_REV', 'FIRST3_BOE', 'FIRST3_LIQ', 'FIRST3_GAS', 'LAT_LENGTH', 'WELL_CNT',
    #                        'FIRST3REV_PER_FT']].to_excel(writer, 'First3M(' + str(Min_Well) + '+) btwn ' + date_period_12)
    #
    # data_sort_API_OPER_5Plus_6_6[['Operator Name','FIRST6_REV', 'FIRST6_BOE', 'FIRST6_LIQ', 'FIRST6_GAS', 'LAT_LENGTH', 'WELL_CNT',
    #                     'FIRST6REV_PER_FT']].to_excel(writer, 'First6M(' + str(Min_Well) + '+) btwn ' + date_period_6)
    # data_sort_API_OPER_5Plus_3_6[['Operator Name','FIRST3_REV', 'FIRST3_BOE', 'FIRST3_LIQ', 'FIRST3_GAS', 'LAT_LENGTH', 'WELL_CNT',
    #                     'FIRST3REV_PER_FT']].to_excel(writer, 'First3M(' + str(Min_Well) + '+) btwn ' + date_period_6)

    data_sort_API_OPER_5Plus_3_3[['Operator Name','FIRST3_REV', 'FIRST3_BOE', 'FIRST3_LIQ', 'FIRST3_GAS', 'LAT_LENGTH', 'WELL_CNT',
                        'FIRST3REV_PER_FT']].to_excel(writer, 'First3M(' + str(Min_Well) + '+) btwn ' + date_period_3)

    # data_sort_API_OPER_12[['FIRST12_REV', 'FIRST12_BOE', 'FIRST12_LIQ', 'FIRST12_GAS', 'LAT_LENGTH', 'WELL_CNT',
    #                     'FIRST12REV_PER_FT']].to_excel(writer, 'First12M btwn ' + date_period_12)
    # data_sort_API_OPER_12[['FIRST6_REV', 'FIRST6_BOE', 'FIRST6_LIQ', 'FIRST6_GAS', 'LAT_LENGTH', 'WELL_CNT',
    #                     'FIRST6REV_PER_FT']].to_excel(writer, 'First6M btwn ' + date_period_12)
    # data_sort_API_OPER_12[['FIRST3_REV', 'FIRST3_BOE', 'FIRST3_LIQ', 'FIRST3_GAS', 'LAT_LENGTH', 'WELL_CNT',
    #                     'FIRST3REV_PER_FT']].to_excel(writer, 'First3M btwn ' + date_period_12)
    #
    # data_sort_API_OPER_6[['FIRST6_REV', 'FIRST6_BOE', 'FIRST6_LIQ', 'FIRST6_GAS', 'LAT_LENGTH', 'WELL_CNT',
    #                     'FIRST6REV_PER_FT']].to_excel(writer, 'First6M btwn ' + date_period_6)
    # data_sort_API_OPER_6[['FIRST3_REV', 'FIRST3_BOE', 'FIRST3_LIQ', 'FIRST3_GAS', 'LAT_LENGTH', 'WELL_CNT',
    #                     'FIRST3REV_PER_FT']].to_excel(writer, 'First3M btwn ' + date_period_6)

    data_sort_API_OPER_3[['FIRST3_REV', 'FIRST3_BOE', 'FIRST3_LIQ', 'FIRST3_GAS', 'LAT_LENGTH', 'WELL_CNT',
                        'FIRST3REV_PER_FT']].to_excel(writer, 'First3M btwn ' + date_period_3)

    writer.save()
    writer.close()

    print('Done')


if __name__ == '__main__':

    main()



