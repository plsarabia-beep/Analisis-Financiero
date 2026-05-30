
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Financial Analysis Dashboard")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Configuration Panel")

tickers = st.sidebar.text_input("Tickers", "AAPL,MSFT,GOOGL")
benchmark = st.sidebar.text_input("Benchmark", "^GSPC")

start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("today"))

rf = st.sidebar.number_input("Tasa libre de riesgo (%)", value=5.0)/100
capital = st.sidebar.number_input("Capital", value=100000)
alpha = st.sidebar.number_input("Nivel significancia α", value=0.05)
horizon = st.sidebar.number_input("Plazo (días)", value=1)

confidence = 1 - alpha
z_value = norm.ppf(confidence)

symbols = [t.strip() for t in tickers.split(",")]

# =========================
# DATA
# =========================
data = yf.download(symbols + [benchmark], start=start, end=end)["Adj Close"]
returns = data.pct_change().dropna()
benchmark_returns = returns[benchmark]

# =========================
# FUNCIONES
# =========================
def annualized_return(r):
    return (1 + r.mean())**252 - 1

def annualized_volatility(r):
    return r.std() * np.sqrt(252)

def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return (cum - peak).min()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Benchmark", "Matrices", "CAPM Avanzado"]
)

# =========================
# TAB 1: METRICAS
# =========================
with tab1:
    st.subheader("Indicadores completos")

    metrics = pd.DataFrame()

    for col in symbols:
        r = returns[col]

        beta = np.cov(r, benchmark_returns)[0][1] / np.var(benchmark_returns)
        corr = np.corrcoef(r, benchmark_returns)[0][1]

        ann_return = annualized_return(r)
        ann_vol = annualized_volatility(r)

        sharpe = (ann_return - rf) / ann_vol if ann_vol != 0 else np.nan
        treynor = (ann_return - rf) / beta if beta != 0 else np.nan
        capm = rf + beta * (annualized_return(benchmark_returns) - rf)

        var = z_value * r.std() * np.sqrt(horizon) * capital

        metrics.loc[col, "Rentabilidad anualizada"] = ann_return
        metrics.loc[col, "Volatilidad anualizada"] = ann_vol
        metrics.loc[col, "Sharpe"] = sharpe
        metrics.loc[col, "Treynor"] = treynor
        metrics.loc[col, "Beta"] = beta
        metrics.loc[col, "Correlación"] = corr
        metrics.loc[col, "CAPM"] = capm
        metrics.loc[col, "VaR"] = var
        metrics.loc[col, "VaR %"] = var / capital

    st.dataframe(metrics.style.format("{:.2%}"))

# =========================
# TAB 2: BENCHMARK
# =========================
with tab2:
    cum = (1 + returns).cumprod()
    st.line_chart(cum)

# =========================
# TAB 3: MATRICES
# =========================
with tab3:
    st.subheader("Correlation")
    st.dataframe(returns.corr())

    st.subheader("Covariance")
    st.dataframe(returns.cov())

# =========================
# TAB 4: CAPM REGRESION
# =========================
with tab4:
    st.subheader("CAPM con OLS")

    capm_results = pd.DataFrame()

    for col in symbols:

        r_i = returns[col] - rf/252
        r_m = benchmark_returns - rf/252

        X = sm.add_constant(r_m)
        model = sm.OLS(r_i, X).fit()

        capm_results.loc[col, "Alpha"] = model.params["const"] * 252
        capm_results.loc[col, "Beta"] = model.params[benchmark]
        capm_results.loc[col, "p-value Beta"] = model.pvalues[benchmark]
        capm_results.loc[col, "R²"] = model.rsquared

    st.dataframe(capm_results)

    # gráfico
    asset = st.selectbox("Activo", symbols)

    r_i = returns[asset]
    r_m = benchmark_returns

    model = sm.OLS(r_i, sm.add_constant(r_m)).fit()
    alpha = model.params["const"]
    beta = model.params[benchmark]

    x = np.linspace(r_m.min(), r_m.max(), 100)
    y = alpha + beta * x

    fig, ax = plt.subplots()
    ax.scatter(r_m, r_i, alpha=0.3)
    ax.plot(x, y, color="red")
    st.pyplot(fig)
