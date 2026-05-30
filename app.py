# ==========================================
# DASHBOARD FINANCIERO PROFESIONAL COMPLETO
# ==========================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Dashboard Financiero Profesional")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Configuración")

tickers = st.sidebar.text_input("Activos", "AAPL,MSFT,GOOGL")
benchmark = st.sidebar.text_input("Benchmark", "^GSPC")

start = st.sidebar.date_input("Inicio", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("Fin", pd.to_datetime("today"))

rf = st.sidebar.number_input("Tasa libre (%)", 5.0)/100
capital = st.sidebar.number_input("Capital", 100000)
alpha = st.sidebar.number_input("Nivel significancia α", 0.05)
horizon = st.sidebar.number_input("Plazo (días)", 1)

confidence = 1 - alpha
z_value = norm.ppf(confidence)

symbols = [x.strip() for x in tickers.split(",")]

# ==========================================
# DESCARGA ROBUSTA
# ==========================================
try:
    raw = yf.download(symbols + [benchmark], start=start, end=end)

    if raw.empty:
        st.error("No hay datos, revisa tickers")
        st.stop()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.levels[0]:
            data = raw["Adj Close"]
        else:
            data = raw["Close"]
    else:
        if "Adj Close" in raw.columns:
            data = raw["Adj Close"]
        else:
            data = raw["Close"]

except:
    st.error("Error descargando datos")
    st.stop()

if benchmark not in data.columns:
    st.error("Benchmark inválido")
    st.stop()

returns = data.pct_change().dropna()
benchmark_returns = returns[benchmark]

# ==========================================
# FUNCIONES
# ==========================================
def ann_return(r):
    return (1 + r.mean())**252 - 1

def ann_vol(r):
    return r.std() * np.sqrt(252)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Indicadores", "Benchmark", "Matrices", "CAPM Regresión"]
)

# ==========================================
# INDICADORES
# ==========================================
with tab1:

    st.subheader("Indicadores por activo")

    df = pd.DataFrame()

    for asset in symbols:

        if asset not in returns.columns:
            continue

        r = returns[asset]

        mean_d = r.mean()
        vol_d = r.std()

        r_ann = ann_return(r)
        vol_ann = ann_vol(r)

        beta = np.cov(r, benchmark_returns)[0][1] / np.var(benchmark_returns)
        corr = np.corrcoef(r, benchmark_returns)[0][1]

        sharpe = (r_ann - rf) / vol_ann if vol_ann != 0 else np.nan
        treynor = (r_ann - rf) / beta if beta != 0 else np.nan

        capm = rf + beta * (ann_return(benchmark_returns) - rf)

        var = z_value * vol_d * np.sqrt(horizon) * capital
        var_pct = var / capital

        df.loc[asset, "Rentabilidad diaria"] = mean_d
        df.loc[asset, "Volatilidad diaria"] = vol_d
        df.loc[asset, "Rentabilidad anualizada"] = r_ann
        df.loc[asset, "Volatilidad anualizada"] = vol_ann
        df.loc[asset, "iSharpe"] = sharpe
        df.loc[asset, "Coef. Correlación Pearson"] = corr
        df.loc[asset, "BETA"] = beta
        df.loc[asset, "iTraynor"] = treynor
        df.loc[asset, "CAPM"] = capm
        df.loc[asset, "Tasa Libre de Riesgo"] = rf
        df.loc[asset, "Capital"] = capital
        df.loc[asset, "Intervalo Confianza"] = confidence
        df.loc[asset, "Nivel Significancia"] = alpha
        df.loc[asset, "Valor Z"] = z_value
        df.loc[asset, "Plazo"] = horizon
        df.loc[asset, "VaR"] = var
        df.loc[asset, "VaR %"] = var_pct

    st.dataframe(df.style.format({
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
# BENCHMARK
# ==========================================
with tab2:
    st.subheader("Rendimientos acumulados")
    cum = (1 + returns).cumprod()
    st.line_chart(cum)

# ==========================================
# MATRICES
# ==========================================
with tab3:
    st.subheader("Correlación")
    st.dataframe(returns.corr())

    st.subheader("Covarianza")
    st.dataframe(returns.cov())

# ==========================================
# CAPM REGRESIÓN
# ==========================================
with tab4:

    st.subheader("CAPM OLS")

    capm_df = pd.DataFrame()

    for asset in symbols:

        if asset not in returns.columns:
            continue

        r_i = returns[asset] - rf/252
        r_m = benchmark_returns - rf/252

        model = sm.OLS(r_i, sm.add_constant(r_m)).fit()

        capm_df.loc[asset, "Alpha"] = model.params["const"] * 252
        capm_df.loc[asset, "Beta"] = model.params[benchmark]
        capm_df.loc[asset, "p-value"] = model.pvalues[benchmark]
        capm_df.loc[asset, "t-stat"] = model.tvalues[benchmark]
        capm_df.loc[asset, "R²"] = model.rsquared

    st.dataframe(capm_df.style.format({
        "Alpha": "{:.2%}",
        "Beta": "{:.2f}",
        "p-value": "{:.4f}",
        "t-stat": "{:.2f}",
        "R²": "{:.2f}"
    }))

    # gráfico
    st.subheader("Gráfico CAPM")

    asset = st.selectbox("Activo", symbols)

    if asset in returns.columns:

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
