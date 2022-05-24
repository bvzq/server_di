#!pip install pytrends
import pandas as pd
import matplotlib.pyplot as plt
#from pandas_profiling import ProfileReport
from scipy.stats import spearmanr

from datetime import date
from datetime import datetime, timedelta
from time import sleep

from pytrends.request import TrendReq

pytrend = TrendReq(hl='pt-PT', tz=360)

import warnings
warnings.filterwarnings('ignore')

# documentação GTrends https://support.google.com/trends/answer/4365533?hl=pt

cidades = [['Aveiro'],['Acores'],['Beja'],['Braga'],['Braganca'],['Castelo Branco'],['Coimbra'],['Faro'],['Guarda'],
['Leiria'],['Lisboa'],['Madeira'],['Portalegre'],['Porto'],['Santarem'],['Setubal'],['Viana do Castelo'],['Vila Real'],
['Viseu'],['Evora']]

kw_list = [['sintoma'],['farmácia'],['dor'],['constipação'],['falta de ar'],['centro de saúde'],['hospital'],['SNS24'],
           ['urgencias'],['medicamentos'],['febre'],['tosse']] 
kw_list = [['medicina tradicional chinesa']]

# kw_list = [['mascara'] constipado
# demostrar como a palavra mascara é só significativa na etapa post covid
data = pd.DataFrame([])
data0 = pd.DataFrame([])
data_3dias = pd.DataFrame([])
data_5dias = pd.DataFrame([])
data_fim = pd.DataFrame([])
dicc = {}

# today_date = '2019/01/01'
# today_date = datetime.strptime(today_date, '%Y/%m/%d')
# start_date = today_date + timedelta(days=0) 

for i in range(len(kw_list)): 
    
    dia = 0
    start_date = '2021/01/05'
    start_date = datetime.strptime(start_date, '%Y/%m/%d')
    today_date = start_date
    
    data = pd.DataFrame([])
    data0 = pd.DataFrame([])
    data_3dias = pd.DataFrame([])
    data_5dias = pd.DataFrame([])
    data_fim = pd.DataFrame([])
    colunas = []
    colunas_3d = []
    colunas_5d = []
    
    for name in range(len(cidades)):
    
        colunas.append(cidades[name][0] + '_' + kw_list[i][0])
        colunas_3d.append(cidades[name][0] + '_3d_' + kw_list[i][0])
        colunas_5d.append(cidades[name][0] + '_5d_' + kw_list[i][0])
    print("colunas", colunas)
        
    while  (today_date.year == 2021): #  and today_date.month <= 2  and today_date.day <= 10
        
        today_date = start_date + timedelta(days = dia)
        
        today_date1 = today_date 
        
        today_date2 = today_date - timedelta(days = 1)  
        
        today_date3 = today_date  - timedelta(days = 2)
        
        today_date4 = today_date - timedelta(days = 3)
        
        today_date5 = today_date - timedelta(days = 4)
        
        franja_horaria_today = today_date1.isoformat()[0:10] + " " + today_date1.isoformat()[0:10]
        franja_horaria_3d = today_date3.isoformat()[0:10] + " " + today_date1.isoformat()[0:10]
        franja_horaria_5d = today_date5.isoformat()[0:10] + " " + today_date1.isoformat()[0:10]
        
        print(franja_horaria_today, " ", kw_list[i])
        print(franja_horaria_3d, " ", kw_list[i])
        print(franja_horaria_5d, " ", kw_list[i])
        
        try:
            print("try")
            pytrend.build_payload(kw_list = kw_list[i], timeframe = franja_horaria_today, geo = 'PT') #, cat = 419
            data_today = pytrend.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=False)
            print("today")
            pytrend.build_payload(kw_list = kw_list[i], timeframe = franja_horaria_3d, geo = 'PT')
            data_3d = pytrend.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=False)
            print("3d")
            pytrend.build_payload(kw_list = kw_list[i], timeframe = franja_horaria_5d, geo = 'PT')
            data_5d = pytrend.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=False)
            print("5d")
            
        except:
            print("Except")
            sleep(5)
            
            try:
                print("try")
                pytrend.build_payload(kw_list = kw_list[i],  timeframe = franja_horaria_today, geo = 'PT')
                data_today = pytrend.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=False)
                print("today")
                pytrend.build_payload(kw_list = kw_list[i], timeframe = franja_horaria_3d, geo = 'PT')
                data_3d = pytrend.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=False)
                print("3d")
                pytrend.build_payload(kw_list = kw_list[i], timeframe = franja_horaria_5d, geo = 'PT')
                data_5d = pytrend.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=False)
                print("5d")
            
            except:
                zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                cid = pd.DataFrame(cidades)
                zeros = pd.DataFrame(zeros)
                data_today = pd.concat([cid, zeros], axis = 1)
                
                data_today.columns = [kw_list[i][0], franja_horaria_today[0:10]]
                data_today = data_today.set_index(kw_list[i][0])
                
                data_3d = data_today
                data_5d = data_today
                
                print("Except2")
                print("today_data", data_today.head(2))
                pass
        
        #data_today = data_today[data_today.index == 'Lisboa']
        data_today = data_today.transpose()
        data_today.index = [franja_horaria_today[0:10]]
        data_today.columns = colunas

        data_3d = data_3d.transpose()
        data_3d.index = [franja_horaria_today[0:10]]
        data_3d.columns = colunas_3d

        data_5d = data_5d.transpose()
        data_5d.index = [franja_horaria_today[0:10]]
        data_5d.columns = colunas_5d

        print("data_today_index", data_today.index)
        print("data_3d", data_3d.index)
        print("data_5d", data_5d.index)
                
        data0 = pd.concat([data0, data_today], axis = 0)
        data_3dias = pd.concat([data_3dias, data_3d], axis = 0)
        data_5dias = pd.concat([data_5dias, data_5d], axis = 0)
        print("\nindex data today", data.index)
        sleep(7)
        dia += 1
    
    
    data = pd.concat([data0, data_3dias, data_5dias], axis = 1)
    data.to_csv('../saidas/pesquisas_palavras_' + kw_list[i][0] + '.txt')
        
        

