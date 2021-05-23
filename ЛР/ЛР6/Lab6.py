import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn.metrics import *
import matplotlib.pyplot as plt

def get_arr(results, metr, y):
    arr = []
    for i in results:
        metric = get_scorer(metr)
        arr.append(metric(i, y))
    return arr

@st.cache
def load_data(n_class = 10):
    digits = load_digits(n_class=n_class)
    digits_df = pd.DataFrame(data=np.c_[digits['data'], digits['target']], columns=digits['feature_names'] + ['target'])
    return digits_df, digits


st.header('Вывод данных и графиков')

# Параметры моделей
n_class = st.sidebar.slider("n_class", 2, 20, value=10)

# Данные
data_load_state = st.text('Загрузка данных...')
data_df, data = load_data(n_class)
data_load_state.text('Данные загружены!')

st.subheader('Первые 5 значений')
st.write(data_df.head())

if st.checkbox('Показать все данные'):
    st.subheader('Данные')
    st.write(data)

# Разделение выборки
test_size = st.sidebar.slider("test_size", 0.1, 0.9, value=0.3)
digits_x_train, digits_x_test, digits_y_train, digits_y_test = train_test_split(data.data, data.target, test_size=test_size, random_state=1)

# Параметры моделей
n_estimators = st.sidebar.slider("n_estimators", 1, 15, value=5)

#  Обучение моделей
br = BaggingRegressor(n_estimators=n_estimators, random_state=10)
br.fit(digits_x_train, digits_y_train)
br_res = br.predict(digits_x_test)

adb = AdaBoostRegressor(n_estimators=n_estimators, random_state=10)
adb.fit(digits_x_train, digits_y_train)
adb_res = adb.predict(digits_x_test)

ext = ExtraTreesRegressor(n_estimators=n_estimators, random_state=10)
ext.fit(digits_x_train, digits_y_train)
ext_res = ext.predict(digits_x_test)

# Отображение моделей
results = [br_res, adb_res, ext_res]
models = [br, adb, ext]
mod = [i.__class__.__name__ for i in models]

modelss = st.sidebar.multiselect(
    "Choose algorithms", mod
)


metr = [mean_absolute_error, mean_squared_error, median_absolute_error]
metrs = [i.__name__ for i in metr]

n_metr = st.sidebar.selectbox("Метрики", metrs)

if len(modelss) != 0 and n_metr != "":
    m_arr = []
    for i in metr:
        if(i.__name__ == n_metr):
            m_arr = get_arr(results[0:len(modelss)], i, digits_y_test)
    mod = []
    for i in modelss:
        mod.append(i)
    # st.write(mod)
    # st.write(m_arr)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(np.array(mod),np.array(m_arr) , align='center')
    ax.set_title(n_metr)

    pos = np.arange(len(mod))
    for a,b in zip(pos, m_arr):
        ax.text(0.1, a-0.1, str(round(b,3)), color='white')
    # plt.show()
    st.pyplot(fig)