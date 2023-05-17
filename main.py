import pandas as pd
from fbprophet import Prophet


# Classe para armazenar os dados
class DadosMeteorologicos:
    def __init__(self, data, temperatura_ar):
        self.data = data
        self.temperatura_ar = temperatura_ar


# Função para processar o arquivo CSV e retornar uma lista de objetos DadosMeteorologicos
def process_csv(file_path):
    # Lê o arquivo CSV
    df = pd.read_csv(file_path)

    # Lista para armazenar os objetos DadosMeteorologicos
    dados_meteorologicos = []

    # Itera sobre as linhas do dataframe
    for index, row in df.iterrows():
        # Cria um objeto DadosMeteorologicos com os valores da linha atual
        dados = DadosMeteorologicos(
            data=row['Data'],
            temperatura_ar=row['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)']
        )

        # Adiciona o objeto à lista
        dados_meteorologicos.append(dados)

    # Retorna a lista de objetos DadosMeteorologicos
    return dados_meteorologicos


# Função para realizar a previsão dos próximos 5 dias usando o Prophet
def fazer_previsao(dados):
    # Cria um dataframe com os dados no formato esperado pelo Prophet
    df = pd.DataFrame({
        'ds': [dado.data for dado in dados],
        'y': [dado.temperatura_ar for dado in dados]
    })

    # Cria um modelo Prophet
    model = Prophet()

    # Treina o modelo com os dados
    model.fit(df)

    # Gera um dataframe com as datas dos próximos 5 dias
    future_dates = model.make_future_dataframe(periods=5, freq='D')

    # Realiza a previsão para os próximos 5 dias
    forecast = model.predict(future_dates)

    # Retorna o dataframe de previsão
    return forecast[['ds', 'yhat']].tail(5)


# Caminho para o arquivo CSV
caminho_arquivo_csv = 'INMET_SE_RJ_A625_TRES RIOS_01-01-2023_A_31-03-2023.CSV'

# Processa o arquivo CSV e obtém os dados
dados = process_csv(caminho_arquivo_csv)

# Realiza a previsão dos próximos 5 dias
previsao = fazer_previsao(dados)

# Exibe a previsão dos próximos 5 dias
print(previsao)
