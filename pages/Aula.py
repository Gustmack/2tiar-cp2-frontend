import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from pycaret.classification import *

# Configuração inicial da página
st.set_page_config(page_title='Simulador - Case Ifood',
                   page_icon='./images/logo_fiap.png',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.title('Simulador - Conversão de Vendas')

# Descrição do App
with st.expander('Descrição do App', expanded=False):
    st.write('O objetivo principal deste app é .....')

# Sidebar com informações e escolha do tipo de entrada
with st.sidebar:
    c1, c2 = st.columns(2)
    c1.image('./images/logo_fiap.png', width=100)
    c2.write('')
    c2.subheader('Auto ML - Fiap [v1]')

    database = st.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'))

# Abas principais
tab1, tab2 = st.tabs(["Predições", "Análise Detalhada"])

with tab1:
    if database == 'CSV':
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')
        if file:
            Xtest = pd.read_csv(file)
            mdl_rf = load_model('./pickle/pickle_rf_pycaret')
            ypred = predict_model(mdl_rf, data=Xtest, raw_score=True)

            with st.expander('Visualizar CSV carregado:', expanded=False):
                qtd_linhas = st.slider('Visualizar quantas linhas do CSV:',
                                       min_value=5,
                                       max_value=Xtest.shape[0],
                                       step=10,
                                       value=5)
                st.dataframe(Xtest.head(qtd_linhas))

            with st.expander('Visualizar Predições:', expanded=True):
                threshold = st.slider('Threshold (ponto de corte para considerar predição como True)',
                                      min_value=0.0,
                                      max_value=1.0,
                                      step=0.1,
                                      value=0.5)
                Xtest['Predicted_Class'] = (ypred['prediction_score_1'] > threshold).astype(int)
                qtd_true = Xtest[Xtest['Predicted_Class'] == 1].shape[0]
                qtd_false = Xtest[Xtest['Predicted_Class'] == 0].shape[0]

                st.metric('Qtd clientes True', value=qtd_true)
                st.metric('Qtd clientes False', value=qtd_false)

                def color_pred(val):
                    color = 'olive' if val > threshold else 'orangered'
                    return f'background-color: {color}'

                df_view = pd.DataFrame({'prediction_score_1': ypred['prediction_score_1'], 'Predicted_Class': Xtest['Predicted_Class']})
                st.dataframe(df_view.style.applymap(color_pred, subset=['prediction_score_1']))

                csv = df_view.to_csv(sep=';', decimal=',', index=True)
                st.download_button(label='Download CSV',
                                   data=csv,
                                   file_name='Predicoes.csv',
                                   mime='text/csv')
        else:
            st.warning('Arquivo CSV não foi carregado')

with tab2:
    if database == 'CSV' and file:
        st.header("Análise Detalhada das Características dos Clientes")
        threshold = st.slider("Ajuste o Threshold para Análise", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        Xtest['Predicted_Class'] = (ypred['prediction_score_1'] > threshold).astype(int)

        features_to_plot = Xtest.columns.difference(['Predicted_Class', 'prediction_score_1'])
        for feature in features_to_plot:
            fig, ax = plt.subplots()
            sns.boxplot(data=Xtest, x='Predicted_Class', y=feature, ax=ax)
            st.pyplot(fig)
    else:
        st.error('Nenhuma predição disponível para análise. Por favor, carregue e processe um CSV primeiro.')
