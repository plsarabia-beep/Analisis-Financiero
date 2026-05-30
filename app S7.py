# ==========================================
# DASHBOARD FINANCIERO COMPLETO
# ==========================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN
# ==========================================
st.set_page_config(layout="wide")
st.title("📊 Dashboard Financiero Profesional")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Configuración")

tickers = st.sidebar.text_input("Activos (separados por coma)", "AAPL,MSFT,GOOGL")
benchmark = st.sidebar.text_input("Benchmark", "^GSPC")

start = st.sidebar.date_input("Fecha inicio", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("Fecha fin", pd.to_datetime("today"))

rf = st.sidebar.number_input("Tasa libre de riesgo (%)", value=5.0)/100
capital = st.sidebar.number_input("Capital", value=100000)
alpha = st.sidebar.number_input("Nivel significancia α", value=0.05)
horizon = st.sidebar.number_input("Plazo (días)", value=1)

confidence = 1 - alpha
z_value = norm.ppf(confidence)

symbols = [t.strip() for t in tickers.split(",")]

# ==========================================
# DESCARGA DE DATOS
# ==========================================
data = yf.download(symbols + [benchmark], start=start, end=end)["Adj Close"]
returns = data.pct_change().dropna()
benchmark_returns = returns[benchmark]

# ==========================================
# FUNCIONES
# ==========================================
def annualized_return(r):
    return (1 + r.mean())**252 - 1

def annualized_volatility(r):
    return r.std() * np.sqrt(252)

def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return (cum - peak).min()

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Indicadores", "Benchmark", "Matrices", "CAPM (Regresión)"]
)

# ==========================================
# TAB 1: INDICADORES COMPLETOS
# ==========================================
with tab1:
    st.subheader("Indicadores por activo")

    metrics = pd.DataFrame()

    for col in symbols:
        r = returns[col]

        mean_daily = r.mean()
        vol_daily = r.std()

        ann_return = annualized_return(r)
        ann_vol = annualized_volatility(r)

        beta = np.cov(r, benchmark_returns)[0][1] / np.var(benchmark_returns)
        corr = np.corrcoef(r, benchmark_returns)[0][1]

        sharpe = (ann_return - rf) / ann_vol if ann_vol != 0 else np.nan
        treynor = (ann_return - rf) / beta if beta != 0 else np.nan

        capm = rf + beta * (annualized_return(benchmark_returns) - rf)

        var = z_value * vol_daily * np.sqrt(horizon) * capital
        var_pct = var / capital

        metrics.loc[col, "Rentabilidad diaria"] = mean_daily
        metrics.loc[col, "Volatilidad diaria"] = vol_daily
        metrics.loc[col, "Rentabilidad anualizada"] = ann_return
        metrics.loc[col, "Volatilidad anualizada"] = ann_vol
        metrics.loc[col, "iSharpe"] = sharpe
        metrics.loc[col, "Coef. Correlación Pearson"] = corr
        metrics.loc[col, "BETA"] = beta
        metrics.loc[col, "iTraynor"] = treynor
        metrics.loc[col, "CAPM"] = capm
        metrics.loc[col, "Tasa Libre de Riesgo"] = rf
        metrics.loc[col, "Capital"] = capital
        metrics.loc[col, "Intervalo Confianza"] = confidence
        metrics.loc[col, "Nivel Significancia"] = alpha
        metrics.loc[col, "Valor Z"] = z_value
        metrics.loc[col, "Plazo"] = horizon
        metrics.loc[col, "VaR"] = var
        metrics.loc[col, "VaR %"] = var_pct
        metrics.loc[col, "Max Drawdown"] = max_drawdown(r)

    st.dataframe(metrics.style.format({
        "Rentabilidad diaria": "{:.4%}",
        "Volatilidad diaria": "{:.4%}",
        "Rentabilidad anualizada": "{:.2%}",
        "Volatilidad anualizada": "{:.2%}",
        "iSharpe": "{:.2f}",
        "Coef. Correlación Pearson": "{:.2f}",
        "BETA": "{:.2f}",
        "iTraynor": "{:.2f}",
        "CAPM": "{:.2%}",
        "VaR": "{:,.2f}",
        "VaR %": "{:.2%}"
    }))

# ==========================================
# TAB 2: BENCHMARK
# ==========================================
with tab2:
    st.subheader("Rendimientos acumulados vs Benchmark")
    cumulative = (1 + returns).cumprod()
    st.line_chart(cumulative)

# ==========================================
# TAB 3: MATRICES
# ==========================================
with tab3:
    st.subheader("Matriz de Correlación")
    st.dataframe(returns.corr())

    st.subheader("Matriz de Covarianza")
    st.dataframe(returns.cov())

# ==========================================
# TAB 4: CAPM CON REGRESIÓN
# ==========================================
with tab4:
    st.subheader("CAPM - Regresión OLS")

    capm_results = pd.DataFrame()

    for col in symbols:
        r_i = returns[col] - rf/252
        r_m = benchmark_returns - rf/252

        X = sm.add_constant(r_m)
        model = sm.OLS(r_i, X).fit()

        capm_results.loc[col, "Alpha"] = model.params["const"] * 252
        capm_results.loc[col, "Beta"] = model.params[benchmark]
        capm_results.loc[col, "p-value Beta"] = model.pvalues[benchmark]
        capm_results.loc[col, "t-stat Beta"] = model.tvalues[benchmark]
        capm_results.loc[col, "R²"] = model.rsquared

    st.dataframe(capm_results.style.format({
        "Alpha": "{:.2%}",
        "Beta": "{:.2f}",
        "p-value Beta": "{:.4f}",
        "t-stat Beta": "{:.2f}",
        "R²": "{:.2f}"
    }))

    # GRÁFICO CAPM
    st.subheader("Gráfico CAPM")

    asset = st.selectbox("Selecciona activo", symbols)

    r_i = returns[asset]
    r_m = benchmark_returns

    model = sm.OLS(r_i, sm.add_constant(r_m)).fit()
    alpha_val = model.params["const"]
    beta_val = model.params[benchmark]

    x = np.linspace(r_m.min(), r_m.max(), 100)
    y = alpha_val + beta_val * x

    fig, ax = plt.subplots()
    ax.scatter(r_m, r_i, alpha=0.3)
    ax.plot(x, y, color="red")
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Activo")

    st.pyplot(fig)
