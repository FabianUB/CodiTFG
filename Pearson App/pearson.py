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
def loadDataAltTer():
    altTer = pd.read_excel("Dataframes/df_imputedAltTerKNN.xlsx", index_col=0)
    santJoan = calcularAnomaly(altTer, "L17167-72-00001")
    massiesRoda = calcularAnomaly(altTer, "L08116-72-00002")
    
    return santJoan, massiesRoda

@st.cache
def loadDataBaixTer():
    baixTer = pd.read_excel("Dataframes/df_imputedBaixTerKNN.xlsx", index_col=0)
    pasteralCabal = calcularAnomaly(baixTer, "F001242")
    colomers = calcularAnomaly(baixTer, "L17055-72-00002")
    torroellaMontegri = calcularAnomaly(baixTer, "L17199-72-00001")
    
    return pasteralCabal, colomers, torroellaMontegri

@st.cache
def loadDataPrecipitacionsAlt():
    precipitacions = pd.read_csv("finalsDF/DF_SMC.csv", index_col=0)
    #Carregar dades
    santPau = calcularAnomaly(precipitacions, "CI")
    gurb = calcularAnomaly(precipitacions, "V3")
    
    #Moving Average
    santPau['data'] = santPau['data'].rolling(window=5).mean()
    gurb['data'] = santPau['data'].rolling(window=5).mean()
    
    return santPau, gurb

@st.cache
def loadDataPrecipitacionsBaix():
    precipitacions = pd.read_csv("finalsDF/DF_SMC.csv", index_col=0)
    #Carregar dades
    angles = calcularAnomaly(precipitacions, "DN")
    talladaEmporda = calcularAnomaly(precipitacions, "UB")
    
    #Moving Average
    angles['data'] = angles['data'].rolling(window=5).mean()
    talladaEmporda['data'] = talladaEmporda['data'].rolling(window=5).mean()
    
    return angles, talladaEmporda

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

@st.cache()
def getDataRius():
    alt1, alt2 = loadDataAltTer()
    baix1, baix2, baix3 = loadDataBaixTer()
    return alt1, alt2, baix1, baix2, baix3

@st.cache()
def getDataPrecipitacions():
    altPrep1, altPrep2 = loadDataPrecipitacionsAlt()
    baixPrep1, baixPrep2 = loadDataPrecipitacionsBaix()
    return altPrep1, altPrep2, baixPrep1, baixPrep2

alt1, alt2, baix1, baix2, baix3 = getDataRius()
altPrep1, altPrep2, baixPrep1, baixPrep2 = getDataPrecipitacions()

deta = Deta('a0kv6ay3_MBndet58XcwtGCWVntnKqyF743Wcixkt')
series = ["alt1", "alt2", "baix1", "baix2", "baix3", "altPrep1", "altPrep2", "baixPrep1", "baixPrep2"]
conv = {"alt1":alt1, "alt2":alt2, "baix1":baix1, "baix2":baix2, "baix3":baix3, "altPrep1":altPrep1, "altPrep2":altPrep2, "baixPrep1":baixPrep1, "baixPrep2":baixPrep2}
st.sidebar.write('Series a analitzar')
cols = st.sidebar.columns(2)
s1 = cols[0].selectbox("s1", series)
s2 = cols[1].selectbox("s2", series)
st.sidebar.write('Data de inici i de final')
cols = st.sidebar.columns(2)
dateFormat = "%Y-%m-%d"
minDate = datetime.datetime.strptime("2009-01-01", dateFormat)
maxDate = datetime.datetime.strptime("2020-12-31", dateFormat)
value1 = datetime.datetime.strptime("2020-01-01", dateFormat)
value2 = datetime.datetime.strptime("2020-01-31", dateFormat)
inici = st.sidebar.date_input("Data Inici", min_value=minDate, max_value=maxDate, value=value1)
final = st.sidebar.date_input("Data Final", min_value=minDate, max_value=maxDate, value=value2)
st.sidebar.write('Valors de n i m')
cols = st.sidebar.columns(2)
n = cols[0].number_input("n", min_value=0, value=48)
m = cols[1].number_input("m", min_value=0, value=48)
st.sidebar.write('Nombre de resultats pearson Top')
top = st.sidebar.number_input("top", min_value=5)

st.title('Resultats PearsonDF')
tMedian, vMedian, tMean, vMean = pearsonDF(inici, final, conv[s1], conv[s2], n, m, debug=True)
st.write(f"Temps mitjana (median): {tMedian}, Valor mitjana: {vMedian}")
st.write(f"Temps mitja (mean): {tMean}, Valor mitja: {vMean}")
#st.write(f"Temps moda (mode): {tMode}, Valor smitjana: {vMode}")

st.title('Resultats PearsonTop')
pt = pearsonTop(inici, final, conv[s1], conv[s2], top, n, m)
st.dataframe(pt)
topFiveTime = pt.index.to_list()
topFiveValues = pt['r'].to_list()



if st.button('Guardar en BDD'):
    db = deta.Base('TFG')
    row = {'dataInici':str(inici), 'dataFinal':str(final), 'serie1':str(s1), 'serie2':str(s2), 'medianTime':float(tMedian), 'medianValue':float(vMedian), 'meanTime':float(tMean), 'meanValue':float(vMean), 'm':int(m), 'n':int(n),
    'topFiveTime':topFiveTime, 'topFiveValues':topFiveValues}
    db.put(row)
    st.success('Introduida amb exit a la BBDD')




