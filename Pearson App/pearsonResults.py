import streamlit as st
import pandas as pd
import numpy as np
import datetime
from deta import Deta
import itertools
import statistics

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

conv = {"Sant Joan - Massies de Roda":("alt1", "alt2"), "Pasteral - Colomers":("baix1", "baix2"), "Colomers - Toroella":("baix2", "baix3"), "Precipitacions - Sant Joan":("altPrep1", "alt1"), "Precipitacions - Massies de Roda":("altPrep2", "alt2"),
 "Precipitacions - Pasteral":("baixPrep1", "baix1"), "Precipitacions - Colomers":("baixPrep1", "baix2"), "Precipitacions - Colomers (Alt)":("baixPrep2", "baix2"), "Precipitacions - Torroella":("baixPrep2", "baix2")}
series = ["Sant Joan - Massies de Roda", "Pasteral - Colomers", "Colomers - Toroella", "Precipitacions - Sant Joan", "Precipitacions - Massies de Roda", "Precipitacions - Pasteral", "Precipitacions - Colomers", "Precipitacions - Colomers (Alt)", "Precipitacions - Torroella"]

st.write('Series a analitzar')

series = st.selectbox("Series", series)


st.title('Dades Funcions Pearson DF')

df = getDataFromDB()
s1Proc = conv[series][0]
s2Proc = conv[series][1]
df = df.loc[(df['serie1'] == s1Proc) & (df['serie2'] == s2Proc)]
df1 = df.iloc[:,:-2]
st.dataframe(df1)

cols = st.columns(2)

cols[0].metric('Temps segon Mitjana Total', df['medianTime'].median())
cols[1].metric('Temps segons Mitja Total', df['meanTime'].mean())

st.title('Dades Funcions Pearson Top')
df2 = df[['dataInici', 'dataFinal', 'topFiveTime']]
st.dataframe(df2, use_container_width=True)

cols = st.columns(3)

listOfLists = df['topFiveTime'].to_list()
topFive = list(itertools.chain.from_iterable(listOfLists))

modes = statistics.multimode(topFive)
jointList = ','.join(str(x) for x in modes)
cols[0].metric('Temps segon Mitjana (Top 5)', statistics.median(topFive))
cols[1].metric('Temps segons Mitja (Top 5)', statistics.mean(topFive))
cols[2].metric('Temps segons Moda (Top 5)', jointList)










