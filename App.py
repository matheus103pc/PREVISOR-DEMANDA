import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Previsor de Demanda Semanal",
    page_icon="📈",
    layout="wide"
)

# 2. INTERFACE VISUAL - CABEÇALHO
st.title("📈 Previsor de Demanda Semanal")
st.markdown("""
*Este aplicativo foi desenvolvido para apoiar o Planejamento da Produção, transformando dados históricos 
de vendas/pedidos em previsões para as próximas semanas.*
""")
st.divider()

# 3. BARRA LATERAL - ENTRADA DE DADOS
st.sidebar.header("📥 Configurações de Entrada")

# Nome do produto
produto = st.sidebar.text_input("Nome do Produto:", value="Produto A")

# Entrada das demandas históricas
demandas_input = st.sidebar.text_area(
    "Demandas Históricas (separe por vírgula):",
    value="120, 125, 130, 128, 135, 140, 145, 150, 148, 155, 160, 165"
)

# Parâmetros de projeção
semanas_futuras = st.sidebar.slider("Semanas a prever no futuro:", min_value=1, max_value=6, value=4)
alfa = st.sidebar.slider("Coeficiente Alfa (Suavização Exponencial):", min_value=0.1, max_value=0.9, value=0.3, step=0.1)

# 4. PROCESSAMENTO DOS DADOS DE ENTRADA
try:
    # Converte a string de entrada em uma lista de números inteiros
    historico = [float(x.strip()) for x in demandas_input.split(",") if x.strip() != ""]
except ValueError:
    st.error("❌ Erro: Certifique-se de inserir apenas números separados por vírgula.")
    st.stop()

# Validação do tamanho mínimo dos dados
if len(historico) < 4:
    st.warning("⚠️ Dados insuficientes. Insira pelo menos 4 semanas de histórico para cálculos mais precisos.")
    st.stop()

# Validação de valores negativos
if any(n < 0 for n in historico):
    st.error("❌ Erro: A demanda histórica não pode conter valores negativos.")
    st.stop()

# Criando a estrutura temporal do histórico
n_semanas = len(historico)
semanas_historicas = list(range(1, n_semanas + 1))
semanas_futuras_indices = list(range(n_semanas + 1, n_semanas + semanas_futuras + 1))

# 5. CÁLCULO DOS MÉTODOS DE PREVISÃO
previsoes_mm = []
previsoes_se = []
previsoes_rl = []

# --- MÉTODO 1: Média Móvel Simples (Janela de 3 períodos) ---
janela = 3
historico_mm = historico.copy()
for i in range(semanas_futuras):
    # Calcula a média dos últimos 3 valores disponíveis
    valor_mm = np.mean(historico_mm[-janela:])
    previsoes_mm.append(valor_mm)
    historico_mm.append(valor_mm)

# --- MÉTODO 2: Suavização Exponencial Simples ---
historico_se = historico.copy()
for i in range(semanas_futuras):
    # Fórmula: F(t+1) = alfa * D(t) + (1 - alfa) * F(t)
    if i == 0:
        valor_se = (alfa * historico[-1]) + ((1 - alfa) * np.mean(historico))
    else:
        valor_se = (alfa * historico_se[-1]) + ((1 - alfa) * valor_se)
    previsoes_se.append(valor_se)
    historico_se.append(valor_se)

# --- MÉTODO 3: Regressão Linear Simples (Tendência) ---
X = np.array(semanas_historicas).reshape(-1, 1)
y = np.array(historico)
modelo = LinearRegression().fit(X, y)

X_futuro = np.array(semanas_futuras_indices).reshape(-1, 1)
previsoes_rl = modelo.predict(X_futuro).tolist()


# 6. CÁLCULO DE ERRO HISTÓRICO (Média Móvel de 3 semanas simulada no passado)
erros_abs_mm = []
for t in range(janela, n_semanas):
    real = historico[t]
    previsto = np.mean(historico[t-janela:t])
    erros_abs_mm.append(abs(real - previsto))
mae_mm = np.mean(erros_abs_mm) if erros_abs_mm else 0

# Identificação de Tendência pela Regressão Linear (Coeficiente Angular)
coef_tendencia = modelo.coef_[0]
if coef_tendencia > 0.5:
    tendencia = "Crescente"
elif coef_tendencia < -0.5:
    tendencia = "Decrescente"
else:
    tendencia = "Estável"


# 7. EXIBIÇÃO DOS RESULTADOS (DASHBOARD)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Tabelas de Dados")
    
    # Exibir histórico
    df_hist = pd.DataFrame({"Semana": semanas_historicas, "Demanda Real": historico})
    st.write(f"**Histórico de {produto}:**")
    st.dataframe(df_hist.set_index("Semana"), use_container_width=True)
    
    # Exibir Previsões Futuras
    df_prev = pd.DataFrame({
        "Semana": semanas_futuras_indices,
        "Média Móvel": [round(x, 1) for x in previsoes_mm],
        "Suavização Exp.": [round(x, 1) for x in previsoes_se],
        "Regressão Linear": [round(x, 1) for x in previsoes_rl]
    })
    st.write("**Previsões Futuras Calculadas:**")
    st.dataframe(df_prev.set_index("Semana"), use_container_width=True)

with col2:
    st.subheader("📊 Gráfico Comparativo")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot do Histórico
    ax.plot(semanas_historicas, historico, marker='o', label='Histórico Real', color='black', linewidth=2)
    
    # Linhas de previsão unindo o último ponto histórico às projeções
    ultimo_w = semanas_historicas[-1]
    ultimo_v = historico[-1]
    
    ax.plot([ultimo_w] + semanas_futuras_indices, [ultimo_v] + previsoes_mm, marker='x', linestyle='--', label='Média Móvel (3 sem)')
    ax.plot([ultimo_w] + semanas_futuras_indices, [ultimo_v] + previsoes_se, marker='^', linestyle='--', label='Suavização Exponencial')
    ax.plot([ultimo_w] + semanas_futuras_indices, [ultimo_v] + previsoes_rl, marker='s', linestyle='--', label='Regressão Linear (Tendência)')
    
    ax.set_xlabel("Semanas")
    ax.set_ylabel("Quantidade Demandada")
    ax.set_title(f"Previsão de Demanda para: {produto}")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    st.pyplot(fig)
    
    # KPIs Rápidos
    st.subheader("🔍 Indicadores do Modelo")
    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Tendência Calculada", tendencia)
    kpi2.metric("Erro Médio Histórico (MAE Média Móvel)", f"{round(mae_mm, 2)} unid.")

st.divider()

# 8. SEÇÃO: RECOMENDAÇÃO GERENCIAL
st.subheader("💡 Recomendação Gerencial para Tomada de Decisão")

if tendencia == "Crescente":
    st.success(f"""
    📈 **Atenção Planejamento!** A demanda do **{produto}** apresenta uma clara tendência de **CRESCIMENTO**.
    * **Impacto na Produção:** Recomenda-se utilizar as projeções da **Regressão Linear**, pois os métodos de média podem subestimar a necessidade real.
    * **Ação Recomendada:** Avalie a capacidade produtiva das próximas semanas. Há riscos iminentes de **falta de produto** e perda de vendas caso a produção permaneça estática. Considere turnos extras se necessário.
    """)
elif tendencia == "Decrescente":
    st.warning(f"""
    📉 **Atenção Planejamento!** A demanda do **{produto}** apresenta uma tendência de **QUEDA**.
    * **Impacto na Produção:** Continuar produzindo nos níveis anteriores gerará **excesso de estoque**, aumentando drasticamente os custos de armazenagem e riscos de obsolescência.
    * **Ação Recomendada:** Reduza o ritmo de produção gradativamente, acompanhando a projeção da Regressão Linear ou adote a Média Móvel para uma desaceleração mais suave.
    """)
else:
    st.info(f"""
    ⚖️ **Demanda Estável:** O produto **{produto}** apresenta um comportamento linear e **ESTÁVEL**.
    * **Impacto na Produção:** O risco de grandes erros é menor. A **Média Móvel** ou a **Suavização Exponencial** são excelentes referências para manter a estabilidade da fábrica.
    * **Ação Recomendada:** Mantenha os lotes padrão de produção atuais e use a previsão como teto de segurança para compras de matéria-prima.
    """)

st.caption("⚠️ *Lembrete Acadêmico: Previsão de demanda serve como apoio e direcionamento, não representa uma certeza absoluta do mercado.*")
