import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from numba import njit
from window_ops.shift import shift_array
from sklearn.preprocessing import FunctionTransformer
from mlforecast.target_transforms import GlobalSklearnTransformer, BaseTargetTransform
from mlforecast.lag_transforms import RollingMean, RollingMax, RollingMin

from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive

from mlforecast.utils import PredictionIntervals
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse
from statsmodels.tsa.seasonal import STL


st.set_page_config(
    page_title="Jeu de Trading",
    layout="wide"  
)

###### PARAMS ######


HORIZON = 6
METRICS_LIST = [mae, rmse]
INTERVAL_RESCALE = [4,8]
INIT_FOUNDS = 500 
NB_IDS = 4
INIT_MAX_STOCK = 3
INIT_YEARS_PLOT = 3
N_WINDOWS_INTERVAL = 8


###### DATA ######


@st.cache_data
def load_data():
    train = pd.read_csv('data/train_game.csv')
    train['Date'] = pd.to_datetime(train['Date'])
    test = pd.read_csv('data/test_game.csv')
    test['Date'] = pd.to_datetime(test['Date'])
    return train, test

def initialize_session_state(train, INIT_FOUNDS, NB_IDS, INIT_MAX_STOCK, INIT_YEARS_PLOT):
    if 'funds' not in st.session_state:
        st.session_state.funds = INIT_FOUNDS
    if 'selected_ids' not in st.session_state:
        st.session_state.selected_ids = np.random.choice(train['unique_id'].unique(), NB_IDS, replace=False)
    if 'stock' not in st.session_state:
        st.session_state.stock = {id: np.random.randint(0,INIT_MAX_STOCK) for id in st.session_state.selected_ids}
    if 'purchase_price' not in st.session_state:
        st.session_state.purchase_price = {id: np.nan for id in st.session_state.selected_ids}
    if 'predictions' not in st.session_state:
        st.session_state.predictions = pd.DataFrame()
    if 'baseline' not in st.session_state:
        st.session_state.baseline = pd.DataFrame()
    if 'predictions_memories' not in st.session_state:
        st.session_state.predictions_memories = pd.DataFrame()
    if 'first_predictions_bool' not in st.session_state:
        st.session_state.first_predictions_bool = True
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'buy_mode' not in st.session_state:
        st.session_state.buy_mode = False
    if 'sell_mode' not in st.session_state:
        st.session_state.sell_mode = False
    if 'years_plot' not in st.session_state:
        st.session_state.years_plot = INIT_YEARS_PLOT
    if 'shopping_days' not in st.session_state:
        st.session_state.shopping_days = []
    if 'sales_days' not in st.session_state:
        st.session_state.sales_days = []

def reset_session_state(train, INIT_FOUNDS, NB_IDS, INIT_MAX_STOCK, INIT_YEARS_PLOT):
    st.session_state.funds = INIT_FOUNDS
    st.session_state.selected_ids = np.random.choice(train['unique_id'].unique(), NB_IDS, replace=False)
    st.session_state.stock = {id: np.random.randint(0, INIT_MAX_STOCK) for id in st.session_state.selected_ids}
    st.session_state.purchase_price = {id: np.nan for id in st.session_state.selected_ids}
    st.session_state.predictions = pd.DataFrame()
    st.session_state.baseline = pd.DataFrame()
    st.session_state.predictions_memories = pd.DataFrame()
    st.session_state.first_predictions_bool = True
    st.session_state.current_step = 0
    st.session_state.buy_mode = False
    st.session_state.sell_mode = False
    st.session_state.years_plot = INIT_YEARS_PLOT
    st.session_state.shopping_days = []
    st.session_state.sales_days = []


###### MODEL ######


@njit
def diff_over_previous(x, offset):
    return x - shift_array(x, offset=offset)

class LocalMinMaxScaler(BaseTargetTransform):
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.stats_ = df.groupby(self.id_col)[self.target_col].agg(['min', 'max'])
        df = df.merge(self.stats_, on=self.id_col)
        df[self.target_col] = (df[self.target_col] - df['min']) / (df['max'] - df['min'])
        df = df.drop(columns=['min', 'max'])
        return df
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(self.stats_, on=self.id_col)
        for col in df.columns.drop([self.id_col, self.time_col, 'min', 'max']):
            df[col] = df[col] * (df['max'] - df['min']) + df['min']
        df = df.drop(columns=['min', 'max'])
        return df


def prepare_prediction_data(train, test, current_step, HORIZON):
    cutoff_date = train['Date'].max() + timedelta(weeks=HORIZON * (current_step - 1))
    test_cutoff_date = test[test['Date'] <= cutoff_date]
    return pd.concat([train, test_cutoff_date])


@st.cache_data
def make_predictions(data, HORIZON):
    sk_log1p = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
    lgbm_params = {
        'n_estimators': 133, 
        'learning_rate': 0.053112414244801495, 
        'num_leaves': 17, 
        'min_data_in_leaf': 18, 
        'feature_fraction': 0.8591550644702561
    }

    fcst = MLForecast(
        models=[LGBMRegressor(**lgbm_params, verbosity=-1, n_jobs=-1, random_state=42)], 
        freq='W',
        lags=range(1,60),
        date_features=['year', 'month', 'week'],
        target_transforms=[LocalMinMaxScaler(), GlobalSklearnTransformer(sk_log1p)],
        lag_transforms={
            1: [
                RollingMean(window_size=8, min_samples=8),
                RollingMax(window_size=8, min_samples=8),
                RollingMin(window_size=8, min_samples=8),
                RollingMax(window_size=16, min_samples=16),
                RollingMin(window_size=16, min_samples=16),
                (diff_over_previous, 1),
                (diff_over_previous, 6)
            ],
        }
    )
    fcst.fit(
        df=data[['Date', 'unique_id', 'Prix']],
        static_features=[],
        id_col='unique_id',
        time_col='Date',
        target_col='Prix',
        prediction_intervals=PredictionIntervals(n_windows=N_WINDOWS_INTERVAL, h=HORIZON),
    )
    predictions = fcst.predict(HORIZON, level=[95])
    return predictions 

def baseline_model(data, HORIZON):
    fcst = StatsForecast(
        models=[Naive(), SeasonalNaive(season_length=52)],
        freq='W'
    )
    fcst.fit(
        df=data[['Date', 'unique_id', 'Prix']],
        id_col='unique_id',
        time_col='Date',
        target_col='Prix',
    )
    predictions = fcst.predict(h=HORIZON)
    return predictions.reset_index()
    

def evaluate_model(data, predictions, baseline, selected_id, METRICS_LIST):
    data_filtered = data[data['unique_id'].eq(selected_id)].copy()
    predictions_filtered = predictions[predictions['unique_id'].eq(selected_id)].copy()
    join_df = data_filtered.merge(predictions_filtered, on=['unique_id', 'Date'])\
        .merge(baseline, on=['unique_id', 'Date'])

    result = evaluate(
        df = join_df[['unique_id', 'Date', 'Prix', 'Naive', 'SeasonalNaive', 'LGBMRegressor']],
        metrics=METRICS_LIST,
        id_col='unique_id',
        time_col='Date',
        target_col='Prix',
    ).drop(
        columns=['unique_id']
    ).set_index('metric')
    st.table(result.T)


###### RESCALE ######


def rescale_value(value, min_value, max_value, min_new_value, max_new_value):
    numerator = (value - min_value) * (max_new_value - min_new_value)
    denominator = max_value - min_value
    return min_new_value + numerator / denominator

def rescale_col(df, cols, selected_ids, minmax_df, INTERVAL_RESCALE):
    df = df[df['unique_id'].isin(selected_ids)].copy()
    minmax_df = minmax_df.reset_index().copy()
    df = df.merge(minmax_df, on=['unique_id'])
    for col in cols:
        df[col] = df.apply(
            lambda row: rescale_value(row[col], row['Min'], row['Max'], INTERVAL_RESCALE[0], INTERVAL_RESCALE[1]), 
            axis=1
        )
    return df.drop(columns=['Min', 'Max'])


###### PLOT ######


@st.cache_data
def generate_plots(
    data, predictions, past_predictions, selected_ids, current_step, shopping_days=None, 
    sales_days=None, years_plot=4, PLOT_PRED=True
):
    fig = make_subplots(
        rows=len(selected_ids), cols=1,  shared_xaxes=True, vertical_spacing=0.05, 
        subplot_titles=[f"Cotations pour {unique_id}" for unique_id in selected_ids]
    )

    for idx, unique_id in enumerate(selected_ids):
        filtered_train = data[data['unique_id'] == unique_id].copy()
        filtered_train = filtered_train[
            filtered_train['Date'] >= filtered_train['Date'].max() - timedelta(days=years_plot * 365)
        ]
        first_of_years = pd.date_range(
            start=filtered_train['Date'].min(),
            end=filtered_train['Date'].max(),
            freq='YS'
        )

        fig.add_trace(
            go.Scatter(
                x=filtered_train['Date'],
                y=filtered_train['Prix'],
                mode='lines',
                name="Prix",  
                line=dict(color="blue"),
                legendgroup='Prix',
                showlegend=(idx == 0),  
            ), row=idx + 1, col=1,
        )

        if PLOT_PRED:
            filtered_predictions = predictions[predictions['unique_id'] == unique_id].copy()
            fig.add_trace(
                go.Scatter(
                    x=filtered_predictions['Date'],
                    y=filtered_predictions['LGBMRegressor'],
                    mode='lines',
                    name="Prévisions",
                    line=dict(color="red"),
                    legendgroup='Predictions',
                    showlegend=(idx == 0)
                ),
                row=idx + 1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([filtered_predictions['Date'], filtered_predictions['Date'][::-1]]),
                    y=pd.concat([filtered_predictions['LGBMRegressor-hi-95'], filtered_predictions['LGBMRegressor-lo-95'][::-1]]),
                    fill='toself',
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    line=dict(color='rgba(255, 0, 0, 0)'),  # Pas de ligne
                    name="Intervalle avec 95% de chances d'obtenir la valeur réelle",
                    legendgroup='Intervalle de confiance',
                    showlegend=(idx == 0)
                ),
                row=idx + 1, col=1
            )

        if current_step > 1:
            filtered_past_predictions = past_predictions[past_predictions['unique_id'] == unique_id].copy()
            fig.add_trace(
                go.Scatter(
                    x=filtered_past_predictions['Date'],
                    y=filtered_past_predictions['LGBMRegressor'],
                    mode='lines',
                    name="Prévisions passées",
                    line=dict(color='green'),
                    legendgroup='Predictions passées',
                    visible='legendonly' if current_step <=10 else True,
                    showlegend=(idx == 0)
                ),
                row=idx + 1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([filtered_past_predictions['Date'], filtered_past_predictions['Date'][::-1]]),
                    y=pd.concat([filtered_past_predictions['LGBMRegressor-hi-95'], filtered_past_predictions['LGBMRegressor-lo-95'][::-1]]),
                    fill='toself',
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    line=dict(color='rgba(255, 0, 0, 0)'),  # Pas de ligne
                    legendgroup='Intervalle de confiance',
                    showlegend=False
                ),
                row=idx + 1, col=1
            )   

        shopping_days_df = pd.DataFrame(shopping_days, columns=['unique_id', 'Date', 'Prix'])
        sales_days_df = pd.DataFrame(sales_days, columns=['unique_id', 'Date', 'Prix'])
        fig.add_trace(
            go.Scatter(
                x=shopping_days_df[shopping_days_df['unique_id'] == unique_id]['Date'],
                y=shopping_days_df[shopping_days_df['unique_id'] == unique_id]['Prix'],
                mode='markers',
                name="Achat",
                marker=dict(size=10, color='red'),
                legendgroup='Achat',
                showlegend=(idx == 0)
            ),
            row=idx + 1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=sales_days_df[sales_days_df['unique_id'] == unique_id]['Date'],
                y=sales_days_df[sales_days_df['unique_id'] == unique_id]['Prix'],
                mode='markers',
                name="Vente",
                marker=dict(size=10, color='black'),
                legendgroup='Vente',
                showlegend=(idx == 0)
            ),
            row=idx + 1, col=1
        )

        for year_start in first_of_years:
            fig.add_vline(
                x=year_start, line_dash="dot", line_color="gray", row=idx + 1, col=1,
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0.5, xanchor="center",
                y=-0.1, yanchor="top",
                font=dict(size=16)
            ),
            margin=dict(t=20, b=0, l=0, r=0)
        )
    return fig


###### TRADING INTERFACE ######


def manage_trading_actions(data, minmax_df, INTERVAL_RESCALE, mode):
    selected_id = st.selectbox('Sélectionnez un poisson : ', options=st.session_state.selected_ids)
    current_price = np.round(rescale_value(
        data[data['unique_id'] == selected_id]['Prix'].iloc[-1], minmax_df.loc[selected_id,'Min'], 
        minmax_df.loc[selected_id,'Max'], INTERVAL_RESCALE[0], INTERVAL_RESCALE[1]
    ), 2)
    
    st.write(f"Prix : {current_price} €")

    if mode == 'buy':
        max_quantity = int(st.session_state.funds // current_price)
    else:
        max_quantity = st.session_state.stock[selected_id]
    
    if max_quantity > 0:
        if not np.isnan(st.session_state.purchase_price[selected_id]) and mode == 'sell':
            st.write(
                f'Rendement depuis le dernier achat : {np.round((current_price - st.session_state.purchase_price[selected_id]) / st.session_state.purchase_price[selected_id] * 100, 3)} %'
            )

        quantity = st.slider('Choisir la quantité :', min_value=0, max_value=max_quantity)
        total = current_price * quantity
        st.write(f"**Total : {total:.2f} €**")

        if st.button('Confirmer'):
            current_day = data[data['unique_id'] == selected_id]['Date'].iloc[-1]
            if mode == 'buy':
                st.session_state.funds -= total
                st.session_state.stock[selected_id] += quantity
                st.success("Achat confirmé ! 🎉")
                st.session_state.shopping_days.append((selected_id, current_day, current_price))
                st.session_state.purchase_price[selected_id] = current_price
            else:
                st.session_state.funds += total
                st.session_state.stock[selected_id] -= quantity
                st.success("Vente confirmée ! 🎉")
                st.session_state.sales_days.append((selected_id, current_day, current_price))

    elif mode == 'buy':
        st.write(f"Les fonds disponibles ne sont pas suffisants.")
    else:
        st.write(f"L'inventaire de {selected_id} est à 0")


###### ANALYSIS TIME SERIES ######


def statistic_serie(df, predictions, selected_id):
    col1, col2 = st.columns([3,0.5])
    with col1:
        df_filtre = df[df['unique_id'] == selected_id]
        predictions_filtre = predictions[predictions['unique_id'] == selected_id]
        fig_analyse = go.Figure()
        fig_analyse.add_trace(
            go.Scatter(
                x=df_filtre['Date'],
                y=df_filtre['Prix'],
                mode='lines',
                name='Prix Réels',
                line=dict(color='blue')
            )
        )
        
        # Ajout des prédictions
        fig_analyse.add_trace(
            go.Scatter(
                x=predictions_filtre['Date'],
                y=predictions_filtre['LGBMRegressor'],
                mode='lines',
                name='Prédictions',
                line=dict(color='red')
            )
        )
        fig_analyse.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0.5, xanchor="center",
                y=-0.1, yanchor="top",
                font=dict(size=16)
            ),
            margin=dict(t=20, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_analyse)
    with col2:
        st.write('')
        st.write('')
        st.write('')
        st.table(df[df['unique_id'] == selected_id]['Prix'].describe())


def seasonal_analysis(df, selected_id, period=52):
    filtered_df = df[df['unique_id'] == selected_id].copy()

    decomposition = STL(filtered_df['Prix'], period=period).fit()

    filtered_df['Prix sans tendance'] = filtered_df['Prix'] - decomposition.trend
    filtered_df['Mois'] = filtered_df['Date'].dt.month

    fig_box = px.box(
        filtered_df,
        x='Mois', y='Prix sans tendance', color='Mois', points='all',
    )
    fig_box.update_layout(xaxis_title="Mois", yaxis_title="Prix sans tendance")
    st.plotly_chart(fig_box, use_container_width=True)


###### INTERFACE ###### 


train, test = load_data()
minmax_df = pd.concat([train, test]).groupby(['unique_id'])['Prix'].agg(Min='min', Max='max')
initialize_session_state(train, INIT_FOUNDS, NB_IDS, INIT_MAX_STOCK, INIT_YEARS_PLOT)

st.title("📈 Jeu de Trading 📈")

if st.session_state.current_step == 0:
    st.write(
        """
        🌊 **Bienvenue dans l'univers impitoyable des marchés de la pêche !** 🐟

        Prêt(e) à plonger dans l'univers impitoyable… des marchés de la pêche ?
        
        Anticipez les hausses et les baisses pour maximiser vos profits tout en évitant les stocks invendus !

        **Votre mission :**  
        - Anticipez les fluctuations du marché grâce à un modèle de machine learning (LGBM). 
        - Acheter, vendre, et éviter la crise des anchois.  
        - Gérez vos fonds pour rester rentable et dominer le marché.

        **10 étapes pour tout rafler !**  
        Affrontez les vagues avec audace, ou jouez la sécurité pour éviter le naufrage... Tout dépend de vous !
        
        Cliquez sur **"Commencer"** pour relever le défi ! 
        """
    )
    if st.button("Commencer 🐠"):
        st.session_state.current_step += 1
        st.rerun()

elif st.session_state.current_step <= 10 :
    if st.button("Lancer l'étape suivante"):
        st.session_state.current_step += 1
        st.session_state.first_predictions_bool = True
        if st.session_state.current_step <= 10:
            st.session_state.predictions_memories = pd.concat(
                [st.session_state.predictions_memories, st.session_state.predictions], ignore_index=True
            )   
        st.rerun()

    data = prepare_prediction_data(train, test, st.session_state.current_step, HORIZON)
    
    if st.session_state.first_predictions_bool and st.session_state.current_step < 10:    
        st.session_state.predictions = make_predictions(data, HORIZON)
        baseline_df = baseline_model(data[data['unique_id'].isin(st.session_state.selected_ids)], HORIZON)
        st.session_state.baseline = pd.concat(
                [st.session_state.baseline, baseline_df], ignore_index=True
            ) 
    
    st.session_state.years_plot = st.radio('Années à afficher : ', [1,2,3,4], index=st.session_state.years_plot-1,  horizontal=True)

    col1, col2= st.columns([3,1])
    with col1:
        data_rescale = rescale_col(data, ['Prix'], st.session_state.selected_ids, minmax_df, INTERVAL_RESCALE)
        predictions_rescale = rescale_col(
            st.session_state.predictions, ['LGBMRegressor', 'LGBMRegressor-hi-95', 'LGBMRegressor-lo-95'], 
            st.session_state.selected_ids, minmax_df, INTERVAL_RESCALE
        )
        if st.session_state.current_step > 1:
            predictions_memories_rescale = rescale_col(
                st.session_state.predictions_memories, ['LGBMRegressor', 'LGBMRegressor-hi-95', 'LGBMRegressor-lo-95'], 
                st.session_state.selected_ids, minmax_df, INTERVAL_RESCALE
            )
        else:
            predictions_memories_rescale = pd.DataFrame({})

        fig = generate_plots(
            data_rescale, predictions_rescale, predictions_memories_rescale,
            st.session_state.selected_ids, st.session_state.current_step, years_plot=st.session_state.years_plot,
            PLOT_PRED=True if st.session_state.current_step < 10 else False
            )
        st.plotly_chart(fig)
    
        st.session_state.first_predictions_bool = False

    with col2:
        st.markdown("## 🛒 **Actions de Trading**")
        action = st.radio('Actions', ['Acheter', 'Vendre'], index=0)
        if action == 'Acheter':
            manage_trading_actions(data, minmax_df, INTERVAL_RESCALE, mode='buy')
        elif action =='Vendre':
            manage_trading_actions(data, minmax_df, INTERVAL_RESCALE, mode='sell')

else :
    col1_bis, col2_bis, col3_bis= st.columns([0.5,3,0.5])
    with col2_bis:
        st.title('🏆 Fin du Jeu : Vos Résultats !')
        profits = st.session_state.funds - INIT_FOUNDS

        st.write(f"- ##### **Profit total :** {profits:.2f} €")
        
        if profits > INIT_FOUNDS:
            st.subheader("- ##### 🥇 **Titre : Trader des Océans !**")
            st.write("🌊 Vous avez maîtrisé l'art du commerce maritime. Vos prises sont légendaires, et vos comptes brillent ! Bravo !")
        elif profits > INIT_FOUNDS / 2:
            st.write("- ##### 🥈 **Titre : Marchand des Mers !**")
            st.write("Vous avez fait fructifier vos prises et vos ventes avec brio. Un capitaine d'expérience, mais la mer a encore des trésors à offrir ! ⚓🐠")
        else:
            st.write("- ##### 🥉 **Titre : Pêcheur Audacieux !**")
            st.write("🐟 Vos débuts promettent des aventures encore plus lucratives !")

        st.write('')
        st.write("### 🔍 **Visualisez vos décisions de trading**")
        data = prepare_prediction_data(train, test, 10, HORIZON)
        data_rescale = rescale_col(data, ['Prix'], st.session_state.selected_ids, minmax_df, INTERVAL_RESCALE)
        predictions_rescale = rescale_col(
            st.session_state.predictions, ['LGBMRegressor', 'LGBMRegressor-hi-95', 'LGBMRegressor-lo-95'], 
            st.session_state.selected_ids, minmax_df, INTERVAL_RESCALE
        )
        predictions_memories_rescale = rescale_col(
            st.session_state.predictions_memories, ['LGBMRegressor', 'LGBMRegressor-hi-95', 'LGBMRegressor-lo-95'], 
            st.session_state.selected_ids, minmax_df, INTERVAL_RESCALE
        )
        fig = generate_plots(
            data_rescale, predictions_rescale, predictions_memories_rescale, st.session_state.selected_ids, 
            st.session_state.current_step, shopping_days=st.session_state.shopping_days, 
            sales_days=st.session_state.sales_days, years_plot=3, PLOT_PRED=False
        )
        st.plotly_chart(fig)

        st.write("### 🌟 **Envie de retenter votre chance ?**")
        st.write("###### Prenez votre revanche ou perfectionnez votre stratégie. Le marché n'attend que vous !")
    
    if st.button("🔄 Rejouer une partie"):
        reset_session_state(train, INIT_FOUNDS, NB_IDS, INIT_MAX_STOCK, INIT_YEARS_PLOT)
        st.rerun()

    st.divider()
    if st.toggle('🔍 Analyse Serie Temporelle'):
        st.markdown("### 🎣 **Sélectionnez un poisson pour commencer l'analyse**")
        selected_id = st.selectbox(
            'Choisissez un poisson :', 
            options=st.session_state.selected_ids,
            help="Cliquez sur un poisson pour explorer ses prix réels, les prédictions, et des analyses saisonnières."
        )

        tab1, tab2, tab3 = st.tabs(["📈 Statistiques des prix", "🗓️ Analyse saisonnière", "⚙️ Évaluation des modèles"])

        with tab1:
            st.markdown("### 📈 **Statistiques des prix réels et des prédictions**")
            st.markdown("Comparez les prix historiques et les prédictions générées par le modèle.")
            statistic_serie(data, st.session_state.predictions_memories, selected_id)

        with tab2:
            st.markdown("### 🗓️ **Analyse des tendances saisonnières**")
            st.markdown("Découvrez comment les prix varient au fil des mois une fois les tendances générales supprimées.")
            seasonal_analysis(data, selected_id, period=52)

        with tab3:
            st.markdown("### ⚙️ **Évaluation des performances des modèles**")
            st.markdown("Comparez les modèles utilisés pour prédire les prix.")
            col1, col2, col3 = st.columns([1,1,1])
            with col2:
                evaluate_model(data, st.session_state.predictions_memories, st.session_state.baseline, selected_id, METRICS_LIST)

with st.sidebar:
    st.markdown("# ⚡ **Progression**")
    step = min(st.session_state.current_step, 10)
    st.progress(step / 10)
    st.markdown(f"### Étape actuelle : {step} / 10")
    st.divider()

    st.markdown("# 💰 **Fonds disponibles**")
    st.markdown(f"## **{st.session_state.funds:.2f} €**")
    st.divider()
    
    st.markdown("# 📦 **Inventaire**")
    for id_, quantity in st.session_state.stock.items():
        st.markdown(f"- **{id_}** : {quantity} unité{'s' if quantity > 1 else ''}")


