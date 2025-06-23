import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import re

sns.set(style="whitegrid")

# Inicializar session_state si no existe
if 'tabla_resultados_df' not in st.session_state:
    st.session_state['tabla_resultados_df'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'dfv' not in st.session_state:
    st.session_state['dfv'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = "1️⃣ Subida y ajuste"

# Menú lateral que actualiza page
page = st.sidebar.radio(
    "Selecciona la etapa:",
    ["1️⃣ Subida y ajuste", "2️⃣ Simulación base", "3️⃣ Simulación con inflación","4️⃣ Simulación con inflación y fenómenos", "5️⃣ Simulación inflación, fenómenos y recesión","6️⃣ Análisis de escenarios"],
    index=["1️⃣ Subida y ajuste", "2️⃣ Simulación base", "3️⃣ Simulación con inflación","4️⃣ Simulación con inflación y fenómenos", "5️⃣ Simulación inflación, fenómenos y recesión","6️⃣ Análisis de escenarios"].index(st.session_state.page)
)

# Si el usuario selecciona algo distinto, actualizamos y refrescamos
if page != st.session_state.page:
    st.session_state.page = page
    st.rerun()


if page == "1️⃣ Subida y ajuste":
    st.title("Análisis de datos")
    # ---- Subida de archivos ----
    uploaded_file = st.file_uploader("Carga tu archivo con la base de datos de la empresa", type=['xlsx'])
    uploaded_file_v = st.file_uploader("Carga tu archivo con las variables exógenas", type=['xlsx'])

    if uploaded_file and uploaded_file_v:
        # ---- Cargar archivos ----
        df = pd.read_excel(uploaded_file, sheet_name="Base CH")
        dfv = pd.read_excel(uploaded_file_v)
        dfv = dfv.loc[:, ~dfv.columns.str.contains('^Unnamed')]

        # ---- Procesamiento inicial ----
        df['PERIODO'] = df['PERIODO'].astype(str)
        df['AÑO'] = df['AÑO'].astype(str)
        df['MES_AÑO'] = df['PERIODO'] + '-' + df['AÑO']

        # ---- Agrupaciones ----
        ton_mes = df.groupby('MES_AÑO').agg({
            'TONELADAS': 'sum',
            'INGRESO': 'sum',
            'COSTO MATERIA PRIMA': 'sum',
            'OTROS COSTOS': 'sum'
        }).reset_index()
        ton_mes = ton_mes[ton_mes['TONELADAS'] > 0].copy()
        ton_mes['INGRESO_TON_MES'] = ton_mes['INGRESO'] / ton_mes['TONELADAS']
        ton_mes['COSTO_MP_TON_MES'] = ton_mes['COSTO MATERIA PRIMA'] / ton_mes['TONELADAS']
        ton_mes['OTROS_COSTOS_TON_MES'] = ton_mes['OTROS COSTOS'] / ton_mes['TONELADAS']

        ton_marca_mes = df.groupby(['MES_AÑO', 'MARCA']).agg({
            'TONELADAS': 'sum',
            'INGRESO': 'sum',
            'COSTO MATERIA PRIMA': 'sum',
            'OTROS COSTOS': 'sum'
        }).reset_index()
        ton_marca_mes = ton_marca_mes[ton_marca_mes['TONELADAS'] > 0].copy()
        ton_marca_mes['INGRESO_TON_MES'] = ton_marca_mes['INGRESO'] / ton_marca_mes['TONELADAS']
        ton_marca_mes['COSTO_MP_TON_MES'] = ton_marca_mes['COSTO MATERIA PRIMA'] / ton_marca_mes['TONELADAS']
        ton_marca_mes['OTROS_COSTOS_TON_MES'] = ton_marca_mes['OTROS COSTOS'] / ton_marca_mes['TONELADAS']

        # ---- Ajuste de distribuciones ----
        resultados_tabla = []

        def fit_and_record(data, var_name):
            data = data.dropna()
            if len(data) < 5:
                resultados_tabla.append({
                    'Variable': var_name,
                    'Distribución': 'Empírica',
                    'Fórmula': 'Distribución de los datos sin ajuste paramétrico'
                })
                return

            resultados = {}
            formulas = {}

            mu, sigma = stats.norm.fit(data)
            resultados['Normal'] = stats.kstest(data, 'norm', args=(mu, sigma)).statistic
            formulas['Normal'] = f"norm(mu={mu:.2f}, sigma={sigma:.2f})"

            if (data > 0).all():
                s, loc, scale = stats.lognorm.fit(data)
                resultados['Log-normal'] = stats.kstest(data, 'lognorm', args=(s, loc, scale)).statistic
                formulas['Log-normal'] = f"lognorm(s={s:.2f}, loc={loc:.2f}, scale={scale:.2f})"

            loc, scale = stats.expon.fit(data)
            resultados['Exponencial'] = stats.kstest(data, 'expon', args=(loc, scale)).statistic
            formulas['Exponencial'] = f"expon(loc={loc:.2f}, scale={scale:.2f})"

            c = (data.mean() - data.min()) / (data.max() - data.min())
            resultados['Triangular'] = stats.kstest(data, 'triang', args=(c, data.min(), data.max()-data.min())).statistic
            formulas['Triangular'] = f"triang(c={c:.2f}, min={data.min():.2f}, range={data.max()-data.min():.2f})"

            loc, scale = stats.uniform.fit(data)
            resultados['Uniforme'] = stats.kstest(data, 'uniform', args=(loc, scale)).statistic
            formulas['Uniforme'] = f"uniform(loc={loc:.2f}, scale={scale:.2f})"

            norm_data = (data - data.min()) / (data.max() - data.min())
            a, b, locb, scaleb = stats.beta.fit(norm_data)
            resultados['Beta'] = stats.kstest(norm_data, 'beta', args=(a, b, locb, scaleb)).statistic
            formulas['Beta'] = f"beta(a={a:.2f}, b={b:.2f}, loc={locb:.2f}, scale={scaleb:.2f})"

            ag, locg, scaleg = stats.gamma.fit(data)
            resultados['Gamma'] = stats.kstest(data, 'gamma', args=(ag, locg, scaleg)).statistic
            formulas['Gamma'] = f"gamma(a={ag:.2f}, loc={locg:.2f}, scale={scaleg:.2f})"

            cw, locw, scalew = stats.weibull_min.fit(data)
            resultados['Weibull'] = stats.kstest(data, 'weibull_min', args=(cw, locw, scalew)).statistic
            formulas['Weibull'] = f"weibull_min(c={cw:.2f}, loc={locw:.2f}, scale={scalew:.2f})"

            bp, locp, scalep = stats.pareto.fit(data)
            resultados['Pareto'] = stats.kstest(data, 'pareto', args=(bp, locp, scalep)).statistic
            formulas['Pareto'] = f"pareto(b={bp:.2f}, loc={locp:.2f}, scale={scalep:.2f})"

            best_fit = min(resultados, key=resultados.get)
            resultados_tabla.append({
                'Variable': var_name,
                'Distribución': best_fit,
                'Fórmula': formulas[best_fit]
            })

        # Generales
        fit_and_record(ton_mes['TONELADAS'], 'TONELADAS_MES')
        fit_and_record(ton_mes['INGRESO_TON_MES'], 'INGRESO_TON_MES')
        fit_and_record(ton_mes['COSTO_MP_TON_MES'], 'COSTO_MP_TON_MES')
        fit_and_record(ton_mes['OTROS_COSTOS_TON_MES'], 'OTROS_COSTOS_TON_MES')

        # Marcas y Submarcas
        for marca in ton_marca_mes['MARCA'].unique():
            subset = ton_marca_mes[ton_marca_mes['MARCA'] == marca]
            fit_and_record(subset['TONELADAS'], f'TONELADAS_MES_MARCA_{marca}')
            fit_and_record(subset['INGRESO_TON_MES'], f'INGRESO_TON_MES_MARCA_{marca}')
            fit_and_record(subset['COSTO_MP_TON_MES'], f'COSTO_MP_TON_MES_MARCA_{marca}')
            fit_and_record(subset['OTROS_COSTOS_TON_MES'], f'OTROS_COSTOS_TON_MES_MARCA_{marca}')

        tabla_resultados_df = pd.DataFrame(resultados_tabla)

        # ---- Variables numéricas del segundo archivo ----
        variables = dfv.select_dtypes(include=np.number).columns
        resultados_finales = []

        def fit_distributions(data, var_name):
            data = data.dropna()
            if len(data) < 5:
                st.warning(f"⚠ Muy pocos datos para {var_name}")
                return

            resultados = {}

            mu, sigma = stats.norm.fit(data)
            ks, p = stats.kstest(data, 'norm', args=(mu, sigma))
            resultados['Normal'] = (ks, p, f"norm(mu={mu:.2f}, sigma={sigma:.2f})")

            if (data > 0).all():
                s, loc, scale = stats.lognorm.fit(data)
                ks, p = stats.kstest(data, 'lognorm', args=(s, loc, scale))
                resultados['Log-normal'] = (ks, p, f"lognorm(s={s:.2f}, loc={loc:.2f}, scale={scale:.2f})")

            loc, scale = stats.expon.fit(data)
            ks, p = stats.kstest(data, 'expon', args=(loc, scale))
            resultados['Exponencial'] = (ks, p, f"expon(loc={loc:.2f}, scale={scale:.2f})")

            c = (data.mean() - data.min()) / (data.max() - data.min())
            ks, p = stats.kstest(data, 'triang', args=(c, data.min(), data.max() - data.min()))
            resultados['Triangular'] = (ks, p, f"triang(c={c:.2f}, min={data.min():.2f}, range={data.max()-data.min():.2f})")

            loc, scale = stats.uniform.fit(data)
            ks, p = stats.kstest(data, 'uniform', args=(loc, scale))
            resultados['Uniforme'] = (ks, p, f"uniform(loc={loc:.2f}, scale={scale:.2f})")

            norm_data = (data - data.min()) / (data.max() - data.min())
            a, b, locb, scaleb = stats.beta.fit(norm_data)
            ks, p = stats.kstest(norm_data, 'beta', args=(a, b, locb, scaleb))
            resultados['Beta'] = (ks, p, f"beta(a={a:.2f}, b={b:.2f}, loc={locb:.2f}, scale={scaleb:.2f})")

            ag, locg, scaleg = stats.gamma.fit(data)
            ks, p = stats.kstest(data, 'gamma', args=(ag, locg, scaleg))
            resultados['Gamma'] = (ks, p, f"gamma(a={ag:.2f}, loc={locg:.2f}, scale={scaleg:.2f})")

            cw, locw, scalew = stats.weibull_min.fit(data)
            ks, p = stats.kstest(data, 'weibull_min', args=(cw, locw, scalew))
            resultados['Weibull'] = (ks, p, f"weibull(c={cw:.2f}, loc={locw:.2f}, scale={scalew:.2f})")

            bp, locp, scalep = stats.pareto.fit(data)
            ks, p = stats.kstest(data, 'pareto', args=(bp, locp, scalep))
            resultados['Pareto'] = (ks, p, f"pareto(b={bp:.2f}, loc={locp:.2f}, scale={scalep:.2f})")

            best = min(resultados.items(), key=lambda x: x[1][0])

            resultados_finales.append({
                'Variable': var_name,
                'Distribución': best[0],
                'KS': best[1][0],
                'P-valor': best[1][1],
                'Fórmula': best[1][2]
            })

        for var in variables:
            fit_distributions(dfv[var], var)
        resultados_df = pd.DataFrame(resultados_finales)

        st.subheader("Resultados de ajustes (variables generales)")
        st.dataframe(resultados_df)

        st.subheader("Resultados de ajustes (por marca / tonelada)")
        st.dataframe(tabla_resultados_df)

        # Guarda en session_state
        st.session_state['df'] = df
        st.session_state['dfv'] = dfv
        st.session_state['tabla_resultados_df'] = tabla_resultados_df
        st.session_state['resultados_variables_df'] = resultados_df

        st.success("✅ Las bases de datos y distribuciones quedaron cargadas correctamente.")

        if st.button("Ir a la página 2️⃣ Simulación base"):
            st.session_state.page =  "2️⃣ Simulación base"
            st.rerun()
        
elif page ==  "2️⃣ Simulación base":

    if st.session_state['tabla_resultados_df'] is None or st.session_state['df'] is None:
        st.error("Primero debes realizar el ajuste de distribuciones en la página 1.")
    else:
        df = st.session_state['df']
        tabla_resultados_df = st.session_state['tabla_resultados_df']
        st.title("Análisis de datos y simulación Simple")
        # ---- Simulación ----
        def simular_variable(fila, n_sim, n_meses):
            dist_name = fila['Distribución']
            formula = fila['Fórmula']
            params = dict(re.findall(r'(\w+)=([-\d\.]+)', formula))
            params = {k: float(v) for k, v in params.items()}

            if dist_name == 'Normal':
                return stats.norm.rvs(loc=params['mu'], scale=params['sigma'], size=(n_sim, n_meses))
            elif dist_name == 'Log-normal':
                return stats.lognorm.rvs(s=params['s'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Exponencial':
                return stats.expon.rvs(loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Triangular':
                return stats.triang.rvs(c=params['c'], loc=params['min'], scale=params['range'], size=(n_sim, n_meses))
            elif dist_name == 'Uniforme':
                return stats.uniform.rvs(loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Beta':
                return stats.beta.rvs(a=params['a'], b=params['b'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Gamma':
                return stats.gamma.rvs(a=params['a'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Weibull':
                return stats.weibull_min.rvs(c=params['c'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Pareto':
                return stats.pareto.rvs(b=params['b'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            else:
                raise ValueError(f"Distribución {dist_name} no soportada")

        n_sim = 1000
        n_meses = 7
        meses_nombres = ['Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        marcas = df['MARCA'].unique()

        sim_marca = {}
        for marca in marcas:
            sim_marca[marca] = {}
            for var in ['TONELADAS_MES', 'INGRESO_TON_MES', 'COSTO_MP_TON_MES', 'OTROS_COSTOS_TON_MES']:
                fila = tabla_resultados_df[tabla_resultados_df['Variable'] == f'{var}_MARCA_{marca}'].iloc[0]
                sim_marca[marca][var] = simular_variable(fila, n_sim, n_meses)

        utilidad_marca = {}
        ingreso_marca = {}
        for marca in marcas:
            ingreso = sim_marca[marca]['TONELADAS_MES'] * sim_marca[marca]['INGRESO_TON_MES']
            ingreso_marca[marca] = ingreso
            utilidad_marca[marca] = ingreso - sim_marca[marca]['TONELADAS_MES'] * (
                sim_marca[marca]['COSTO_MP_TON_MES'] + sim_marca[marca]['OTROS_COSTOS_TON_MES']
            )

        resultados_marca_mes = []
        for marca in marcas:
            for i, mes in enumerate(meses_nombres):
                utilidad_mes = utilidad_marca[marca][:, i]
                ingreso_mes = ingreso_marca[marca][:, i]
                margen = np.where(ingreso_mes > 0, utilidad_mes / ingreso_mes, 0)
                resultados_marca_mes.append({
                    'Marca': marca,
                    'Mes': mes,
                    'Promedio': utilidad_mes.mean(),
                    'Mediana': np.median(utilidad_mes),
                    'P5': np.percentile(utilidad_mes, 5),
                    'P95': np.percentile(utilidad_mes, 95),
                    'Margen Bruto %': margen.mean() * 100
                })

        tabla_marca_mes_df = pd.DataFrame(resultados_marca_mes)
        st.subheader("Resultados de la simulación por marca y mes")
        st.dataframe(tabla_marca_mes_df)

        utilidad_total_sim = sum(utilidad_marca.values())
        ingreso_total_sim = sum(ingreso_marca.values())

        st.subheader("Distribución simulada de utilidad total (7 meses)")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.hist(utilidad_total_sim.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
        ax.axvline(np.percentile(utilidad_total_sim.sum(axis=1), 5), color='red', linestyle='--', label='P5')
        ax.axvline(np.median(utilidad_total_sim.sum(axis=1)), color='green', linestyle='-', label='Mediana')
        ax.axvline(np.percentile(utilidad_total_sim.sum(axis=1), 95), color='orange', linestyle='--', label='P95')
        ax.set_xlabel('Utilidad total')
        ax.set_ylabel('Frecuencia')
        ax.legend()
        st.pyplot(fig)

        resultados_global_mes = []
        for i, mes in enumerate(meses_nombres):
            utilidad_mes = utilidad_total_sim[:, i]
            ingreso_mes = ingreso_total_sim[:, i]
            margen = np.where(ingreso_mes > 0, utilidad_mes / ingreso_mes, 0)
            resultados_global_mes.append({
                'Mes': mes,
                'Promedio': utilidad_mes.mean(),
                'Mediana': np.median(utilidad_mes),
                'P5': np.percentile(utilidad_mes, 5),
                'P95': np.percentile(utilidad_mes, 95),
                'Margen Bruto %': margen.mean() * 100
            })

        tabla_global_mes_df = pd.DataFrame(resultados_global_mes)
        st.subheader("Resultados globales por mes")
        st.dataframe(tabla_global_mes_df)
        # Guarda los resultados globales para el análisis comparativo
        st.session_state['resultados_simulacion_base'] = tabla_global_mes_df


    if st.button("Ir a la página 3️⃣ Simulación con inflación"):
            st.session_state.page = "3️⃣ Simulación con inflación"
            st.rerun()

elif st.session_state.page == "3️⃣ Simulación con inflación":
    st.header("3️⃣ Simulación con inflación")

    if (st.session_state.tabla_resultados_df is None or 
        st.session_state.resultados_variables_df is None or 
        st.session_state.df is None):
        st.error("Primero debes completar las etapas anteriores (ajuste y simulación base).")
    else:
        df = st.session_state.df
        tabla_resultados_df = st.session_state.tabla_resultados_df
        resultados_variables_df = st.session_state.resultados_variables_df

        n_sim = 1000
        n_meses = 7
        meses_nombres = ['Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        marcas = df['MARCA'].unique()

        def simular_variable(fila, n_sim, n_meses):
            dist_name = fila['Distribución']
            formula = fila['Fórmula']
            params = dict(re.findall(r'(\w+)=([-\d\.]+)', formula))
            params = {k: float(v) for k, v in params.items()}

            if dist_name == 'Normal':
                return stats.norm.rvs(loc=params['mu'], scale=params['sigma'], size=(n_sim, n_meses))
            elif dist_name == 'Log-normal':
                return stats.lognorm.rvs(s=params['s'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Exponencial':
                return stats.expon.rvs(loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Triangular':
                return stats.triang.rvs(c=params['c'], loc=params['min'], scale=params['range'], size=(n_sim, n_meses))
            elif dist_name == 'Uniforme':
                return stats.uniform.rvs(loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Beta':
                return stats.beta.rvs(a=params['a'], b=params['b'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Gamma':
                return stats.gamma.rvs(a=params['a'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Weibull':
                return stats.weibull_min.rvs(c=params['c'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            elif dist_name == 'Pareto':
                return stats.pareto.rvs(b=params['b'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
            else:
                raise ValueError(f"Distribución {dist_name} no soportada")

        # Inflación
        fila_inflacion = resultados_variables_df[resultados_variables_df['Variable'] == 'Inflación mensual (%)'].iloc[0]
        inflacion_sim = simular_variable(fila_inflacion, n_sim, n_meses) / 100

        # Simulación por marca
        sim_marca = {}
        for marca in marcas:
            sim_marca[marca] = {}
            for var in ['TONELADAS_MES', 'INGRESO_TON_MES', 'COSTO_MP_TON_MES', 'OTROS_COSTOS_TON_MES']:
                fila = tabla_resultados_df[tabla_resultados_df['Variable'] == f'{var}_MARCA_{marca}'].iloc[0]
                sim_marca[marca][var] = simular_variable(fila, n_sim, n_meses)

        # Ajustar por inflación
        for marca in marcas:
            sim_marca[marca]['COSTO_MP_TON_MES'] *= (1 + inflacion_sim)
            sim_marca[marca]['OTROS_COSTOS_TON_MES'] *= (1 + inflacion_sim)

        # Calcular utilidad e ingreso
        utilidad_marca = {}
        ingreso_marca = {}
        for marca in marcas:
            ingreso = sim_marca[marca]['TONELADAS_MES'] * sim_marca[marca]['INGRESO_TON_MES']
            ingreso_marca[marca] = ingreso
            utilidad_marca[marca] = ingreso - sim_marca[marca]['TONELADAS_MES'] * (
                sim_marca[marca]['COSTO_MP_TON_MES'] + sim_marca[marca]['OTROS_COSTOS_TON_MES']
            )

        resultados_marca_mes = []
        for marca in marcas:
            for i, mes in enumerate(meses_nombres):
                utilidad_mes = utilidad_marca[marca][:, i]
                ingreso_mes = ingreso_marca[marca][:, i]
                margen = np.where(ingreso_mes > 0, utilidad_mes / ingreso_mes, 0)
                resultados_marca_mes.append({
                    'Marca': marca,
                    'Mes': mes,
                    'Promedio': utilidad_mes.mean(),
                    'Mediana': np.median(utilidad_mes),
                    'P5': np.percentile(utilidad_mes, 5),
                    'P95': np.percentile(utilidad_mes, 95),
                    'Margen Bruto %': margen.mean() * 100
                })

        tabla_marca_mes_df = pd.DataFrame(resultados_marca_mes)
        st.subheader("Resultados por marca y mes con inflación")
        st.dataframe(tabla_marca_mes_df)

        utilidad_total_sim = sum(utilidad_marca.values())
        ingreso_total_sim = sum(ingreso_marca.values())

        st.subheader("Distribución simulada de utilidad total (7 meses)")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.hist(utilidad_total_sim.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
        ax.axvline(np.percentile(utilidad_total_sim.sum(axis=1), 5), color='red', linestyle='--', label='P5')
        ax.axvline(np.median(utilidad_total_sim.sum(axis=1)), color='green', linestyle='-', label='Mediana')
        ax.axvline(np.percentile(utilidad_total_sim.sum(axis=1), 95), color='orange', linestyle='--', label='P95')
        ax.set_xlabel('Utilidad total')
        ax.set_ylabel('Frecuencia')
        ax.legend()
        st.pyplot(fig)

        resultados_global_mes = []
        for i, mes in enumerate(meses_nombres):
            utilidad_mes = utilidad_total_sim[:, i]
            ingreso_mes = ingreso_total_sim[:, i]
            margen = np.where(ingreso_mes > 0, utilidad_mes / ingreso_mes, 0)
            resultados_global_mes.append({
                'Mes': mes,
                'Promedio': utilidad_mes.mean(),
                'Mediana': np.median(utilidad_mes),
                'P5': np.percentile(utilidad_mes, 5),
                'P95': np.percentile(utilidad_mes, 95),
                'Margen Bruto %': margen.mean() * 100
            })

        tabla_global_mes_df = pd.DataFrame(resultados_global_mes)
        st.subheader("Resultados globales por mes con inflación")
        st.dataframe(tabla_global_mes_df)
        # Guarda resultados
        st.session_state['resultados_simulacion_inflacion'] = tabla_global_mes_df

    if st.button("Ir a la página 4️⃣ Simulación con inflación y fenómenos"):
                st.session_state.page = "4️⃣ Simulación con inflación y fenómenos"
                st.rerun()

elif st.session_state.page == "4️⃣ Simulación con inflación y fenómenos":
    st.header("4️⃣ Simulación con inflación y fenómenos")

    if (st.session_state.tabla_resultados_df is None or 
        st.session_state.resultados_variables_df is None or 
        st.session_state.df is None):
        st.error("Primero debes completar las etapas anteriores.")
    else:
        df = st.session_state.df
        tabla_resultados_df = st.session_state.tabla_resultados_df
        resultados_variables_df = st.session_state.resultados_variables_df

        # Buscar inflación y SPEI
        filtro_inflacion = resultados_variables_df[resultados_variables_df['Variable'] == 'Inflación mensual (%)']
        filtro_spei = resultados_variables_df[resultados_variables_df['Variable'] == 'Indicador SPEI']

        if filtro_inflacion.empty or filtro_spei.empty:
            st.error("❌ No se encontraron los datos necesarios de 'Inflación mensual (%)' o 'Indicador SPEI'.")
        else:
            fila_inflacion = filtro_inflacion.iloc[0]
            fila_spei = filtro_spei.iloc[0]

            # Configuración
            n_sim = 1000
            n_meses = 7
            meses_nombres = ['Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
            marcas = df['MARCA'].unique()

            def simular_variable(fila, n_sim, n_meses):
                dist_name = fila['Distribución']
                formula = fila['Fórmula']
                params = dict(re.findall(r'(\w+)=([-\d\.]+)', formula))
                params = {k: float(v) for k, v in params.items()}

                if dist_name == 'Normal':
                    return stats.norm.rvs(loc=params['mu'], scale=params['sigma'], size=(n_sim, n_meses))
                elif dist_name == 'Log-normal':
                    return stats.lognorm.rvs(s=params['s'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Exponencial':
                    return stats.expon.rvs(loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Triangular':
                    return stats.triang.rvs(c=params['c'], loc=params['min'], scale=params['range'], size=(n_sim, n_meses))
                elif dist_name == 'Uniforme':
                    return stats.uniform.rvs(loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Beta':
                    return stats.beta.rvs(a=params['a'], b=params['b'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Gamma':
                    return stats.gamma.rvs(a=params['a'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Weibull':
                    return stats.weibull_min.rvs(c=params['c'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Pareto':
                    return stats.pareto.rvs(b=params['b'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                else:
                    raise ValueError(f"Distribución {dist_name} no soportada")

            inflacion_sim = simular_variable(fila_inflacion, n_sim, n_meses) / 100
            spei_sim = simular_variable(fila_spei, n_sim, n_meses)

            # Simulación por marca
            sim_marca = {}
            for marca in marcas:
                sim_marca[marca] = {}
                for var in ['TONELADAS_MES', 'INGRESO_TON_MES', 'COSTO_MP_TON_MES', 'OTROS_COSTOS_TON_MES']:
                    fila = tabla_resultados_df[tabla_resultados_df['Variable'] == f'{var}_MARCA_{marca}'].iloc[0]
                    sim_marca[marca][var] = simular_variable(fila, n_sim, n_meses)

            # Ajustes por inflación y fenómenos
            def ajustar_por_spei(precio_base, spei):
                factor = np.ones_like(spei)
                factor[spei < -1] = 1.20  # Sequía: +20%
                factor[spei > 1] = 1.15   # Inundación: +15%
                return precio_base * factor

            for marca in marcas:
                sim_marca[marca]['COSTO_MP_TON_MES'] *= (1 + inflacion_sim)
                sim_marca[marca]['OTROS_COSTOS_TON_MES'] *= (1 + inflacion_sim)

                sim_marca[marca]['COSTO_MP_TON_MES'] = ajustar_por_spei(sim_marca[marca]['COSTO_MP_TON_MES'], spei_sim)
                sim_marca[marca]['OTROS_COSTOS_TON_MES'] = ajustar_por_spei(sim_marca[marca]['OTROS_COSTOS_TON_MES'], spei_sim)

            # Cálculo utilidad
            utilidad_marca = {}
            ingreso_marca = {}
            for marca in marcas:
                ingreso = sim_marca[marca]['TONELADAS_MES'] * sim_marca[marca]['INGRESO_TON_MES']
                ingreso_marca[marca] = ingreso
                utilidad_marca[marca] = ingreso - sim_marca[marca]['TONELADAS_MES'] * (
                    sim_marca[marca]['COSTO_MP_TON_MES'] + sim_marca[marca]['OTROS_COSTOS_TON_MES']
                )

            # Resultados por marca
            resultados_marca_mes = []
            for marca in marcas:
                for i, mes in enumerate(meses_nombres):
                    utilidad_mes = utilidad_marca[marca][:, i]
                    ingreso_mes = ingreso_marca[marca][:, i]
                    margen = np.where(ingreso_mes > 0, utilidad_mes / ingreso_mes, 0)
                    resultados_marca_mes.append({
                        'Marca': marca,
                        'Mes': mes,
                        'Promedio': utilidad_mes.mean(),
                        'Mediana': np.median(utilidad_mes),
                        'P5': np.percentile(utilidad_mes, 5),
                        'P95': np.percentile(utilidad_mes, 95),
                        'Margen Bruto %': margen.mean() * 100
                    })

            tabla_marca_mes_df = pd.DataFrame(resultados_marca_mes)
            st.subheader("Resultados por marca y mes")
            st.dataframe(tabla_marca_mes_df)

            # Consolidado
            utilidad_total_sim = sum(utilidad_marca.values())
            ingreso_total_sim = sum(ingreso_marca.values())

            st.subheader("Distribución simulada de utilidad total (7 meses)")
            fig, ax = plt.subplots(figsize=(10,6))
            ax.hist(utilidad_total_sim.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
            ax.axvline(np.percentile(utilidad_total_sim.sum(axis=1), 5), color='red', linestyle='--', label='P5')
            ax.axvline(np.median(utilidad_total_sim.sum(axis=1)), color='green', linestyle='-', label='Mediana')
            ax.axvline(np.percentile(utilidad_total_sim.sum(axis=1), 95), color='orange', linestyle='--', label='P95')
            ax.set_xlabel('Utilidad total')
            ax.set_ylabel('Frecuencia')
            ax.legend()
            st.pyplot(fig)

            # Resultados globales
            resultados_global_mes = []
            for i, mes in enumerate(meses_nombres):
                utilidad_mes = utilidad_total_sim[:, i]
                ingreso_mes = ingreso_total_sim[:, i]
                margen = np.where(ingreso_mes > 0, utilidad_mes / ingreso_mes, 0)
                resultados_global_mes.append({
                    'Mes': mes,
                    'Promedio': utilidad_mes.mean(),
                    'Mediana': np.median(utilidad_mes),
                    'P5': np.percentile(utilidad_mes, 5),
                    'P95': np.percentile(utilidad_mes, 95),
                    'Margen Bruto %': margen.mean() * 100
                })

            tabla_global_mes_df = pd.DataFrame(resultados_global_mes)
            st.subheader("Resultados globales por mes")
            st.dataframe(tabla_global_mes_df)
            st.session_state['resultados_simulacion_inflacion_fenomenos'] = tabla_global_mes_df

    if st.button("Ir a la página 5️⃣ Simulación inflación, fenómenos y recesión"):
                st.session_state.page = "5️⃣ Simulación inflación, fenómenos y recesión"
                st.rerun()

elif st.session_state.page == "5️⃣ Simulación inflación, fenómenos y recesión":
    st.header("5️⃣ Simulación inflación, fenómenos y recesión")

    if (st.session_state.tabla_resultados_df is None or 
        st.session_state.resultados_variables_df is None or 
        st.session_state.df is None):
        st.error("Primero debes completar las etapas anteriores.")
    else:
        df = st.session_state.df
        tabla_resultados_df = st.session_state.tabla_resultados_df
        resultados_variables_df = st.session_state.resultados_variables_df

        # Buscar variables
        f_inflacion = resultados_variables_df[resultados_variables_df['Variable'] == 'Inflación mensual (%)']
        f_spei = resultados_variables_df[resultados_variables_df['Variable'] == 'Indicador SPEI']
        f_recesion = resultados_variables_df[resultados_variables_df['Variable'] == 'Sahm Rule Recession Indicator']

        if f_inflacion.empty or f_spei.empty or f_recesion.empty:
            st.error("❌ Faltan datos de inflación, SPEI o recesión en los resultados.")
        else:
            fila_inflacion = f_inflacion.iloc[0]
            fila_spei = f_spei.iloc[0]
            fila_recesion = f_recesion.iloc[0]

            # Configuración
            n_sim = 1000
            n_meses = 7
            meses_nombres = ['Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
            marcas = df['MARCA'].unique()
            def simular_variable(fila, n_sim, n_meses):
                dist_name = fila['Distribución']
                formula = fila['Fórmula']
                params = dict(re.findall(r'(\w+)=([-\d\.]+)', formula))
                params = {k: float(v) for k, v in params.items()}

                if dist_name == 'Normal':
                    return stats.norm.rvs(loc=params['mu'], scale=params['sigma'], size=(n_sim, n_meses))
                elif dist_name == 'Log-normal':
                    return stats.lognorm.rvs(s=params['s'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Exponencial':
                    return stats.expon.rvs(loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Triangular':
                    return stats.triang.rvs(c=params['c'], loc=params['min'], scale=params['range'], size=(n_sim, n_meses))
                elif dist_name == 'Uniforme':
                    return stats.uniform.rvs(loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Beta':
                    return stats.beta.rvs(a=params['a'], b=params['b'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Gamma':
                    return stats.gamma.rvs(a=params['a'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Weibull':
                    return stats.weibull_min.rvs(c=params['c'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                elif dist_name == 'Pareto':
                    return stats.pareto.rvs(b=params['b'], loc=params['loc'], scale=params['scale'], size=(n_sim, n_meses))
                else:
                    raise ValueError(f"Distribución {dist_name} no soportada")

            # Simulaciones
            inflacion_sim = simular_variable(fila_inflacion, n_sim, n_meses) / 100
            spei_sim = simular_variable(fila_spei, n_sim, n_meses)
            recesion_sim = simular_variable(fila_recesion, n_sim, n_meses)

            # Ajuste funciones
            def ajustar_por_spei(precio_base, spei):
                factor = np.ones_like(spei)
                factor[spei < -1] = 1.20
                factor[spei > 1] = 1.15
                return precio_base * factor

            def ajustar_por_recesion(valor, recesion, tipo):
                factor = np.ones_like(recesion)
                if tipo == 'ingreso':
                    factor[recesion > 0.5] = 0.90
                    factor[(recesion > 0.3) & (recesion <= 0.5)] = 0.95
                elif tipo == 'tonelada':
                    factor[recesion > 0.5] = 0.85
                    factor[(recesion > 0.3) & (recesion <= 0.5)] = 0.92
                elif tipo == 'costo':
                    factor[recesion > 0.5] = 1.05
                    factor[(recesion > 0.3) & (recesion <= 0.5)] = 1.02
                return valor * factor

            # Simulación por marca
            sim_marca = {}
            for marca in marcas:
                sim_marca[marca] = {}
                for var in ['TONELADAS_MES', 'INGRESO_TON_MES', 'COSTO_MP_TON_MES', 'OTROS_COSTOS_TON_MES']:
                    fila = tabla_resultados_df[tabla_resultados_df['Variable'] == f'{var}_MARCA_{marca}'].iloc[0]
                    sim_marca[marca][var] = simular_variable(fila, n_sim, n_meses)

                sim_marca[marca]['COSTO_MP_TON_MES'] *= (1 + inflacion_sim)
                sim_marca[marca]['OTROS_COSTOS_TON_MES'] *= (1 + inflacion_sim)

                sim_marca[marca]['COSTO_MP_TON_MES'] = ajustar_por_spei(sim_marca[marca]['COSTO_MP_TON_MES'], spei_sim)
                sim_marca[marca]['OTROS_COSTOS_TON_MES'] = ajustar_por_spei(sim_marca[marca]['OTROS_COSTOS_TON_MES'], spei_sim)

                sim_marca[marca]['INGRESO_TON_MES'] = ajustar_por_recesion(sim_marca[marca]['INGRESO_TON_MES'], recesion_sim, 'ingreso')
                sim_marca[marca]['TONELADAS_MES'] = ajustar_por_recesion(sim_marca[marca]['TONELADAS_MES'], recesion_sim, 'tonelada')
                sim_marca[marca]['COSTO_MP_TON_MES'] = ajustar_por_recesion(sim_marca[marca]['COSTO_MP_TON_MES'], recesion_sim, 'costo')
                sim_marca[marca]['OTROS_COSTOS_TON_MES'] = ajustar_por_recesion(sim_marca[marca]['OTROS_COSTOS_TON_MES'], recesion_sim, 'costo')

            # Calcular utilidad
            utilidad_marca = {}
            ingreso_marca = {}
            for marca in marcas:
                ingreso = sim_marca[marca]['TONELADAS_MES'] * sim_marca[marca]['INGRESO_TON_MES']
                ingreso_marca[marca] = ingreso
                utilidad_marca[marca] = ingreso - sim_marca[marca]['TONELADAS_MES'] * (
                    sim_marca[marca]['COSTO_MP_TON_MES'] + sim_marca[marca]['OTROS_COSTOS_TON_MES']
                )

            # Resultados por marca
            resultados_marca_mes = []
            for marca in marcas:
                for i, mes in enumerate(meses_nombres):
                    utilidad_mes = utilidad_marca[marca][:, i]
                    ingreso_mes = ingreso_marca[marca][:, i]
                    margen = np.where(ingreso_mes > 0, utilidad_mes / ingreso_mes, 0)
                    resultados_marca_mes.append({
                        'Marca': marca,
                        'Mes': mes,
                        'Promedio': utilidad_mes.mean(),
                        'Mediana': np.median(utilidad_mes),
                        'P5': np.percentile(utilidad_mes, 5),
                        'P95': np.percentile(utilidad_mes, 95),
                        'Margen Bruto %': margen.mean() * 100
                    })
            tabla_marca_mes_df = pd.DataFrame(resultados_marca_mes)
            st.subheader("Resultados por marca y mes")
            st.dataframe(tabla_marca_mes_df)

            # Consolidado
            utilidad_total_sim = sum(utilidad_marca.values())
            ingreso_total_sim = sum(ingreso_marca.values())

            st.subheader("Distribución simulada de utilidad total (7 meses)")
            fig, ax = plt.subplots(figsize=(10,6))
            ax.hist(utilidad_total_sim.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
            ax.axvline(np.percentile(utilidad_total_sim.sum(axis=1), 5), color='red', linestyle='--', label='P5')
            ax.axvline(np.median(utilidad_total_sim.sum(axis=1)), color='green', linestyle='-', label='Mediana')
            ax.axvline(np.percentile(utilidad_total_sim.sum(axis=1), 95), color='orange', linestyle='--', label='P95')
            ax.set_xlabel('Utilidad total')
            ax.set_ylabel('Frecuencia')
            ax.legend()
            st.pyplot(fig)

            resultados_global_mes = []
            for i, mes in enumerate(meses_nombres):
                utilidad_mes = utilidad_total_sim[:, i]
                ingreso_mes = ingreso_total_sim[:, i]
                margen = np.where(ingreso_mes > 0, utilidad_mes / ingreso_mes, 0)
                resultados_global_mes.append({
                    'Mes': mes,
                    'Promedio': utilidad_mes.mean(),
                    'Mediana': np.median(utilidad_mes),
                    'P5': np.percentile(utilidad_mes, 5),
                    'P95': np.percentile(utilidad_mes, 95),
                    'Margen Bruto %': margen.mean() * 100
                })
            tabla_global_mes_df = pd.DataFrame(resultados_global_mes)
            st.subheader("Resultados globales por mes")
            st.dataframe(tabla_global_mes_df)
            st.session_state['resultados_simulacion_inflacion_fenomenos_recesion'] = tabla_global_mes_df

    if st.button("Ir a la página 6️⃣ Análisis de escenarios"):
                st.session_state.page = "6️⃣ Análisis de escenarios"
                st.rerun()

elif st.session_state.page == "6️⃣ Análisis de escenarios":
    st.header("6️⃣ Análisis comparativo de escenarios globales")

    escenarios = {
        "Simulación base": st.session_state.get('resultados_simulacion_base'),
        "Inflación": st.session_state.get('resultados_simulacion_inflacion'),
        "Inflación + fenómenos": st.session_state.get('resultados_simulacion_inflacion_fenomenos'),
        "Inflación + fenómenos + recesión": st.session_state.get('resultados_simulacion_inflacion_fenomenos_recesion')
    }

    # Verifica que todos existan
    missing = [k for k, v in escenarios.items() if v is None]
    if missing:
        st.warning(f"Faltan resultados de: {', '.join(missing)}. Completa esas simulaciones antes de comparar.")
    else:
        # Unifica en un solo DataFrame para Promedio y Margen
        comparativo_promedio = pd.DataFrame()
        comparativo_margen = pd.DataFrame()

        for esc, df in escenarios.items():
            comparativo_promedio[esc] = df.set_index('Mes')['Promedio']
            comparativo_margen[esc] = df.set_index('Mes')['Margen Bruto %']

        # Muestra tabla de promedios
        st.subheader("Promedio de utilidad por mes y escenario")
        st.dataframe(comparativo_promedio.style.format("{:,.0f}"))

        # Muestra tabla de margen
        st.subheader("Margen bruto (%) por mes y escenario")
        st.dataframe(comparativo_margen.style.format("{:.2f}%"))

        # Gráfico de utilidad
        st.subheader("Gráfico: Promedio de utilidad por escenario")
        fig1, ax1 = plt.subplots(figsize=(10,6))
        for esc in comparativo_promedio.columns:
            ax1.plot(comparativo_promedio.index, comparativo_promedio[esc], marker='o', label=esc)
        ax1.set_ylabel("Promedio de utilidad")
        ax1.set_title("Comparativo de promedio de utilidad")
        ax1.legend()
        st.pyplot(fig1)

        # Gráfico de margen
        st.subheader("Gráfico: Margen bruto por escenario")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        for esc in comparativo_margen.columns:
            ax2.plot(comparativo_margen.index, comparativo_margen[esc], marker='o', label=esc)
        ax2.set_ylabel("Margen bruto (%)")
        ax2.set_title("Comparativo de margen bruto")
        ax2.legend()
        st.pyplot(fig2)

        # Resumen ejecutivo
        st.subheader("Resumen ejecutivo 📌")
        st.markdown("""
        - **Simulación base**: Refleja el escenario sin shocks externos, con márgenes más estables.
        - **Inflación**: Se observa un aumento en los costos, lo que reduce el margen bruto en varios meses.
        - **Inflación + fenómenos**: Impacto combinado de inflación y clima adverso, afectando especialmente los meses críticos.
        - **Inflación + fenómenos + recesión**: El escenario más retador, con utilidades promedio más bajas y márgenes significativamente comprimidos en comparación con los otros escenarios.

        Las gráficas ilustran claramente cómo los escenarios externos presionan los resultados financieros, siendo clave para la junta directiva considerar estrategias de mitigación.
        """)
