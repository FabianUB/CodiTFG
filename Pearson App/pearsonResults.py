import streamlit as st
import pandas as pd
import numpy as np
import datetime
from deta import Deta
import itertools
import statistics
from collections import Counter

@st.cache
def getDataFromDB():
    from deta import Deta
    deta = Deta('a0kv6ay3_MBndet58XcwtGCWVntnKqyF743Wcixkt')
    db = deta.Base('TFG')
    res = db.fetch()
    all_items = res.items

    # fetch until last is 'None'
    while res.last:
        res = db.fetch(last=res.last)
        all_items += res.items

    df = pd.DataFrame(all_items)
    df.drop(columns='key', inplace=True)

    return df


st.write('Series a analitzar')

serieFinal = st.selectbox("Serie Final", ['massies', 'colomers'])

if serieFinal == 'massies':
    serieInicial = st.selectbox("Serie Inicial", ['L17147-72-00005','L17079-72-00005', 'F009891', 'alt1', 'DG','CG','CI','V4','CC','V5','CY','VN','WS'])
if serieFinal == 'colomers':
    serieInicial = st.selectbox("Serie Inicial", ['F000005', 'L17079-72-00004', 'F001243', 'KE', 'WS', 'UO', 'UN', 'DJ'])



st.title('Dades Funcions Pearson DF')

df = getDataFromDB()
df = df.loc[(df['serie1'] == str(serieInicial).lower()) & (df['serie2'] == str(serieFinal).lower())]
df1 = df.iloc[:,:-2]
st.dataframe(df1)

cols = st.columns(2)

cols[0].metric('Temps segon Mitjana Total', df['medianTime'].median())
cols[1].metric('Temps segons Mitja Total', df['meanTime'].mean())

st.title('Dades Funcions Pearson Top')
df2 = df[['dataInici', 'dataFinal', 'topFiveTime']]
st.dataframe(df2)

cols = st.columns(2)

listOfLists = df['topFiveTime'].to_list()
c = Counter(itertools.chain.from_iterable(listOfLists))
jointList = c.most_common(5)

values = itertools.chain.from_iterable(listOfLists)



cols[0].metric('Temps segon Mitjana (Top 5)', statistics.median(values))
#cols[1].metric('Temps segons Mitja (Top 5)', statistics.mean(values))

cols = st.columns(5)

cols[0].metric('Moda Top 1', jointList[0][0], jointList[0][1])
cols[1].metric('Moda Top 2', jointList[1][0], jointList[1][1])
cols[2].metric('Moda Top 3', jointList[2][0], jointList[2][1])
cols[3].metric('Moda Top 4', jointList[3][0], jointList[3][1])
cols[4].metric('Moda Top 5', jointList[4][0], jointList[4][1])










