import streamlit as st
import pandas as pd
import numpy as np
import datetime
from deta import Deta
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from statistics import median, mode

@st.cache
def calcularAnomaly(df, col, cont=0.005):
    #Crear una versio horaria del dataset
    df['timestamp'] = pd.to_datetime(df.index)
    df = df.set_index('timestamp').resample("H").mean().reset_index()
    
    #Usem el metode de Isolation Forest per trobar anomalies a les precipitacions
    output = df[[col, 'timestamp']]
    temp = df[[col, 'timestamp']]
    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=cont,max_features=1.0)
    model.fit(temp[[col]].values)
    output['outliers']=pd.Series(model.predict(temp[[col]].values)).apply(lambda x: 'yes' if (x == -1) else 'no' )
    output.rename(columns={col:'data'}, inplace=True)
    
    return output

@st.cache
def loadDataAltTer(nom):
    altTer = pd.read_excel("Dataframes/df_imputedAltTerKNN.xlsx", index_col=0)
    estacio = calcularAnomaly(altTer, nom)
    
    return estacio

@st.cache
def loadDataBaixTer(nom):
    baixTer = pd.read_excel("Dataframes/df_imputedBaixTerKNN.xlsx", index_col=0)
    estacio = calcularAnomaly(baixTer, nom)
    
    return estacio



@st.cache
def loadDataPrecipitacions(nom):
    precipitacions = pd.read_csv("finalsDF/DF_SMC.csv", index_col=0)
    #Carregar dades
    estacio = calcularAnomaly(precipitacions, nom)

    #Moving Average
    estacio['data'] = estacio['data'].rolling(window=5).mean()
    
    
    return estacio

def selectTimespan(df, start_date, end_date):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.loc[(df['timestamp'] >= pd.to_datetime(start_date)) &  (df['timestamp'] <= pd.to_datetime(end_date))].copy(deep=True)
    return df

def pearson_forward_lag(x, y,n=48, m=96, p_ini=0, debug=False):
    timeList = []
    rList = []
    pList = []
    offset = 0
    minutes = 0
    while offset < m:
        r, p = stats.pearsonr(x[p_ini:n+p_ini], y[p_ini+offset:offset+n+p_ini])
        if debug == True:
            print(f"Time Lag: +{minutes}m , Pearson r: {r} and p-value: {p}")
        timeList.append(minutes)
        rList.append(r)
        pList.append(p)
        minutes = minutes + 30
        offset += 1
        
    return timeList, rList, pList

def pearsonDF(inici, final, s1, s2, n=48, m=96, debug=False):
    s1 = selectTimespan(s1, inici, final)
    s2 = selectTimespan(s2, inici, final)
    
    dateFormat = "%Y-%m-%d"
    deltaTime = final - inici
    pointsNum = deltaTime.days * 24 * 2
    times = {}
    canContinue = True
    p_ini = 0
    while canContinue:
        try:
            t, r, p = pearson_forward_lag(s1['data'], s2['data'], n, m, p_ini)
            d = {'r':r, 'p':p}
            df = pd.DataFrame(data=d,index=t)
            
            try:
                temps = df.loc[(df['p'] < 0.05) & (df['r'] > 0.5)]['r'].idxmax()
                valor = df.loc[(df['p'] < 0.05) & (df['r'] > 0.5)]['r'].max()
                
                if debug == True:
                    print(temps, valor)
                times[temps] = valor
            except:
                pass
            if p_ini + n > pointsNum - n:
                canContinue = False
            else:
                p_ini += n
            
            
        except Exception as e:
            if p_ini + n > pointsNum - n:
                canContinue = False
            else:
                p_ini += n
            print("Problema en dateRange: ", inici, " - ", final)
            print(e)
        #times = list(map(int, times.keys()))
        #values = list(map(int, times.keys()))
    return median(times.keys()), median(times.values()), np.array(list(times.keys())).mean(), np.array(list(times.values())).mean()  

def pearsonTop(inici, final, s1, s2, top=5, n=48, m=96):
    s1 = selectTimespan(s1, inici, final)
    s2 = selectTimespan(s2, inici, final)
    
    dateFormat = "%Y-%m-%d"
    deltaTime = final - inici
    pointsNum = deltaTime.days * 24 * 2
    
    t, r, p = pearson_forward_lag(s1['data'], s2['data'])
    
    
    canContinue = True
    p_ini = 0
    
    while canContinue:
        try:
            tNew, rNew, pNew = pearson_forward_lag(s1['data'], s2['data'], n, m, p_ini)
            r += rNew
            p += pNew
            t += tNew
            
            if p_ini + n > pointsNum - n:
                canContinue = False
            else:
                p_ini += n
            
        except Exception as e:
            if p_ini + n > pointsNum - n:
                canContinue = False
            else:
                p_ini += n
            print("Problema en dateRange: ", inici, " - ", final)
            print(e)
    
    d = {'r':r, 'p':p}
    df = pd.DataFrame(data=d,index=t)
    df = df.loc[(df['p'] < 0.05) & (df['r'] > 0.5)]
    df = df.sort_values(by=['r'], ascending=False)
    

    return df.head(top)





massies = loadDataAltTer('L08116-72-00002')
colomers = loadDataBaixTer('L17055-72-00002')
estacionsPlujaAlt = ['DG','CG','CI','V4','CC','V5','CY','VN','WS']
estacionsCabalAlt = ['L17147-72-00005','L17079-72-00005', 'F009891' ]
estacionsPlujaBaix = ['KE', 'WS', 'UO', 'UN', 'DJ']
estacionsCabalBaix = ['F000005', 'L17079-72-00004', 'F001243']

deta = Deta('a0kv6ay3_MBndet58XcwtGCWVntnKqyF743Wcixkt')
final = ['massies', 'colomers']
st.sidebar.write('Series a analitzar')
cols = st.sidebar.columns(2)
s1 = cols[0].selectbox("s1", estacionsCabalBaix)
s2 = cols[1].selectbox("s2", final)
st.sidebar.write('Data de inici i de final')
cols = st.sidebar.columns(2)
fecha = st.selectbox('Fecha Anomalia', [('2010-05-02','2010-05-14'),('2010-10-09','2010-10-16'),('2011-10-09','2011-03-19'),
('2011-11-16','2011-11-23'),('2013-03-05','2013-03-12'),('2013-11-16','2013-11-23'),('2014-11-18','2014-12-04'),('2015-11-01','2015-11-07'),
('2017-02-12','2017-02-26'),('2018-04-08','2018-04-15'),('2019-10-20','2019-10-26'),('2020-01-20','2020-01-27'),
('2020-06-07','2020-06-21'),('2020-08-28','2020-09-03'),('2020-11-26','2020-12-03')])
inici = datetime.datetime.strptime(fecha[0], '%Y-%m-%d')
final = datetime.datetime.strptime(fecha[1], '%Y-%m-%d')
st.sidebar.write('Valors de n i m')
cols = st.sidebar.columns(2)
n = cols[0].number_input("n", min_value=0, value=48)
m = cols[1].number_input("m", min_value=0, value=48)
st.sidebar.write('Nombre de resultats pearson Top')
top = st.sidebar.number_input("top", min_value=5)

if s2 == 'massies':
    st.title('Resultats PearsonDF Massies')

    tMedian, vMedian, tMean, vMean = pearsonDF(inici, final, loadDataPrecipitacions(s1), massies, n, m, debug=True)
    st.write(f"Temps mitjana (median): {tMedian}, Valor mitjana: {vMedian}")
    st.write(f"Temps mitja (mean): {tMean}, Valor mitja: {vMean}")
    #st.write(f"Temps moda (mode): {tMode}, Valor smitjana: {vMode}")

    st.title('Resultats PearsonTop Massies')
    pt = pearsonTop(inici, final, loadDataPrecipitacions(s1), massies, top, n, m)
    st.dataframe(pt)
    topFiveTime = pt.index.to_list()
    topFiveValues = pt['r'].to_list()

elif s2 == 'colomers':
    st.title('Resultats PearsonDF Colomers')

    tMedian, vMedian, tMean, vMean = pearsonDF(inici, final, loadDataBaixTer(s1), colomers, n, m, debug=True)
    st.write(f"Temps mitjana (median): {tMedian}, Valor mitjana: {vMedian}")
    st.write(f"Temps mitja (mean): {tMean}, Valor mitja: {vMean}")
    #st.write(f"Temps moda (mode): {tMode}, Valor smitjana: {vMode}")

    st.title('Resultats PearsonTop Colomers')
    pt = pearsonTop(inici, final, loadDataBaixTer(s1), colomers, top, n, m)
    st.dataframe(pt)
    topFiveTime = pt.index.to_list()
    topFiveValues = pt['r'].to_list()



if st.button('Guardar en BDD'):
    db = deta.Base('TFG')
    row = {'dataInici':str(inici), 'dataFinal':str(final), 'serie1':str(s1).lower(), 'serie2':str(s2).lower(), 'medianTime':float(tMedian), 'medianValue':float(vMedian), 'meanTime':float(tMean), 'meanValue':float(vMean), 'm':int(m), 'n':int(n),
    'topFiveTime':topFiveTime, 'topFiveValues':topFiveValues}
    db.put(row)
    st.success('Introduida amb exit a la BBDD')




