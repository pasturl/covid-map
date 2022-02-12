import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")
mapbox_access_token = st.secrets["TOKEN"]


@st.cache
def filter_data(df, continent):
    df_continent = df[df["continent"] == continent].reset_index(drop=True)
    return df_continent


def filter_data_country(df, country):
    df_country = df[df["location"] == country].reset_index(drop=True)
    return df_country


def agg_data(df_plot, kpi_agg):
    df_total = (df_plot.groupby(["location", "lon", "lat"])
                       .agg({kpi_agg: "max"}).reset_index())
    return df_total


def plot_continent(df, lat_foc, lon_foc, kpi_ts):

    min_deaths = df[kpi_ts].min()
    max_deaths = df[kpi_ts].max()

    fig = px.scatter_geo(df, locations="iso_code", color=kpi_ts,
                         hover_name="location", size=kpi_ts,
                         animation_frame="yw",
                         projection="natural earth",
                         range_color=(min_deaths, max_deaths))

    fig.update_layout(
        width=1200,
        height=700,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        autosize=True,
        geo=dict(
            projection_scale=2,  # this is kind of like zoom
            center=dict(lat=lat_foc, lon=lon_foc),  # this will center on the point
        ))
    return fig


def plot_map(df, mapbox_access_token, lat_foc, lon_foc, kpi_agg):
    scaler = MinMaxScaler(feature_range=(10, 20))
    df[[kpi_agg+'_scale']] = scaler.fit_transform(df[[kpi_agg]])

    fig = go.Figure()

    fig = fig.add_trace(go.Scattermapbox(
        lat=df["lat"].to_numpy(),
        lon=df["lon"].to_numpy(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=df[kpi_agg+'_scale'].to_numpy(), #color=df["location"].to_numpy()
        ), name="Location", text=df[kpi_agg]
    ))

    fig.update_layout(
        width=1200,
        height=700, margin={"r": 0, "t": 0, "l": 0, "b": 0},
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=lat_foc,
                lon=lon_foc
            ),
            pitch=0,
            zoom=2
        ),
    )

    return fig


def plot_ts(df, kpi_ts):
    fig = px.line(df, x='yw', y=kpi_ts)
    fig.update_layout(
        width=1000,
        height=500)
    return fig
# df = read_data()
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df_weekly = pd.read_csv(uploaded_file)
    df_weekly["yw"] = df_weekly["yw"].astype(str)
        
    continents = list(df_weekly.sort_values(by=['continent'])['continent'].unique())


    continent = st.sidebar.selectbox('Continent Selection', continents)
    df_plot = filter_data(df_weekly, continent).reset_index(drop=True)
    locations = list(df_plot.sort_values(by=['location'])['location'].unique())
    country = st.sidebar.selectbox('Country Selection', locations)
    df_country = filter_data_country(df_plot, country).reset_index(drop=True)

    kpis_ts = ['new_cases', 'new_cases_smoothed', 'new_deaths',
               'new_deaths_smoothed', 'new_cases_per_million', 'new_cases_smoothed_per_million',
               'new_deaths_per_million', 'new_deaths_smoothed_per_million', 'hosp_patients',
               'hosp_patients_per_million', 'weekly_icu_admissions',
               'weekly_icu_admissions_per_million', 'new_tests',
               'new_tests_per_thousand', 'positive_rate',
               'people_vaccinated', 'people_fully_vaccinated',
               'new_vaccinations', 'total_vaccinations_per_hundred',
               'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred',
               'stringency_index']
    kpi_ts = st.sidebar.selectbox('KPI Time lapse', kpis_ts)

    kpi_agg = ['total_cases', 'total_deaths','total_cases_per_million',
               'total_deaths_per_million', 'hosp_patients',
               'hosp_patients_per_million', 'weekly_icu_admissions',
               'weekly_icu_admissions_per_million', 'total_tests',
               'total_tests_per_thousand', 'total_vaccinations', 'people_vaccinated',
               'people_fully_vaccinated', 'total_vaccinations_per_hundred',
               'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred',
               'stringency_index']

    kpi_agg = st.sidebar.selectbox('KPI aggregated metric', kpi_agg)


    df_total = agg_data(df_plot, kpi_agg)
    lat_foc = df_plot["lat"].to_numpy().astype(float).mean()
    lon_foc = df_plot["lon"].to_numpy().astype(float).mean()
    fig = plot_continent(df_plot, lat_foc, lon_foc, kpi_ts)
    expander_time = st.expander("Map time-lapse Covid evolution")

    # display streamlit map
    expander_time.plotly_chart(fig)
    my_expander_agg = st.expander("Map aggregated metric")

    fig2 = plot_map(df_total, mapbox_access_token, lat_foc, lon_foc, kpi_agg)
    my_expander_agg.plotly_chart(fig2)

    my_expander_ts = st.expander("Time series chart")
    fig3 = plot_ts(df_country, kpi_ts)
    my_expander_ts.plotly_chart(fig3)


    my_expander_table = st.expander("Summary table")
    # CSS to inject contained in a string
    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
    my_expander_table.dataframe(df_plot)
