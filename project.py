import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


st.title("How are people talking about climate change?")
st.subheader("UNIV102 Let's Talk about Climate Change Final Project")
st.write("Kevin Yin")

st.write("In UNIV102, we often discuss ways in which we communicate climate change and what are the effective ways of discussing its impacts to a potentially non-receptive audience. For my conversation starter, I decided to look at how people have discussed climate change on the largest global forum: Twitter (now X). This project is a series of visualizations that tell the story of climate communication on the internet from a bird's eye view via an exploration in to the Climate Change Twitter Dataset from Effrosynidis et al.")
st.write('Use the sliders to adjust the time window you would like to view. Note that the original dataset spans from 2006 to 2019, but for loading speeds, the dataset has been limited to 2017, which is still 10,000,000 tweets.')
st.markdown('Source: Effrosynidis, Dimitrios, Alexandros I. Karasakalidis, Georgios Sylaios, and Avi Arampatzis. "The climate change Twitter dataset." Expert Systems with Applications 204 (2022): 117541.')
# df = pd.read_csv('The Climate Change Twitter Dataset.csv')
print("DHFLSDHFJLDHFUILDSHFULHSDULGHUISDHGFULHGFDUGS")
df = pd.read_csv('data.csv', engine="pyarrow")
print(df.columns)
df['year'] = pd.DatetimeIndex(df['created_at']).year
df['date'] = pd.DatetimeIndex(df['created_at']).normalize()
df['created_at'] = pd.to_datetime(df['created_at'])


def get_relevant(start = df.date.min(), end = df.date.max()):

    data = df[(df['date'] >= start) & (df['date'] <= end)]
    return data
indexes = df.date.drop_duplicates()
default_range = [indexes.iloc[-30].to_pydatetime(), indexes.iloc[-1].to_pydatetime()]
# values = st.date_input(label="Pick a time range",value=default_range, min_value=df.date.min(), max_value=df.date.max())
values = st.slider(
    "Pick a date range",
    value=default_range,
    format="MM/DD/YY",
    min_value=df.date.min().to_pydatetime(), max_value=df.date.max().to_pydatetime())
print(values)
values = [pd.to_datetime(values[0], utc=True).normalize(), pd.to_datetime(values[1], utc=True).normalize()]
data = get_relevant(values[0], values[1])

def time_series(data):
    sent_group = data[['date', 'sentiment']].groupby(data['date']).mean()
    sent_group["color"] = np.where(sent_group["sentiment"]<0, 'red', 'green')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sent_group.index, y=sent_group['sentiment'], marker_color=sent_group["color"]))

    # Update layout
    fig.update_layout(title='Sentiment Analysis of Tweets',
                    xaxis_title='Date',
                    yaxis_title='Sentiment')
    st.plotly_chart(fig)
time_series(data)
st.write("This is the average daily sentiment of tweents, with sentiment scores ranging from -1 to 1 (from negative to positive respectively). Sentiment and all other annotations were determined by the dataset using various machine learning and AI techniques from classical machine learning to large language models (LLMs).")

def word_count(data):
    counts = data.id.groupby(df['topic']).count()
    go_fig = go.Figure()

    col_obj = go.Scatter(
        x = counts.index, 
        y = counts.values,
        mode = 'markers', 
        marker =dict( size = counts.values/counts.values.max()*100,
        color = [
        "#ff5733",
        "#6a5acd",
        "#1abc9c",
        "#f39c12",
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#9b59b6",
        "#f1c40f",
        "#34495e"
        ]
        ))

    go_fig.add_trace(col_obj)
    go_fig.update_layout(title='Most Discussed Climate Change Topics',
                    xaxis_title='Topics',
                    yaxis_title='Number of Tweets')
    st.plotly_chart(go_fig)
word_count(data)

st.write("As a part of their dataset, Effrosynidis et al. categorize the topics of the tweets into ten categories. The graph shows the frequency of these topics during the specified time range.")

def stance_sentiment(data):
    neut = data[data.stance=='neutral']
    y0 = [neut.sentiment.quantile(1), neut.sentiment.quantile(0), neut.sentiment.quantile(0.25), neut.sentiment.quantile(0.5), neut.sentiment.quantile(0.75)]
    bel = data[data.stance=='believer']
    y1 = [bel.sentiment.quantile(1), bel.sentiment.quantile(0), bel.sentiment.quantile(0.25), bel.sentiment.quantile(0.5), bel.sentiment.quantile(0.75)]
    den = data[data.stance=='denier']
    y2 = [den.sentiment.quantile(1), den.sentiment.quantile(0), den.sentiment.quantile(0.25), den.sentiment.quantile(0.5), den.sentiment.quantile(0.75)]

    fig = go.Figure()
    fig.add_trace(go.Box(y=y0, name="neutral"))
    fig.add_trace(go.Box(y=y1, name="believer"))
    fig.add_trace(go.Box(y=y2, name="denier"))
    fig.update_layout(title='Sentiment per Stance',
                    xaxis_title='Stance',
                    yaxis_title='Sentiment')
    st.plotly_chart(fig)

stance_sentiment(data)
st.write("Effrosynidis et al. also categorize the authors of these tweets into believers, deniers, and neutral. This multi-box plot shows the distribution of sentiment across different stances, with climate change deniers generally being more negative.")
def aggressive(data):
    agg = data.groupby(data['aggressiveness']).count()['id']
    fig = px.pie(agg, values='id', names=agg.index)
    fig.update_layout(title='Ratio of Aggressive Language')
    st.plotly_chart(fig)

aggressive(data)
st.write("Effrosynidis et al. recognize that climate change can be a contentious topic and chose to categorize tweets based on aggressiveness. The above visualization shows the proportion of aggressive to non-aggressive tweets in the given time range, which should be a benchmark to how passionate netizens are about climate change.")
def aggressive_by_stance(data):
    agg = data[data.aggressiveness=='aggressive']
    nonagg = data[data.aggressiveness=='not aggressive']
    agg_st = agg.groupby(agg['stance']).count()['id']
    nonagg_st = nonagg.groupby(nonagg['stance']).count()['id']
    trace = [
        go.Bar(x=agg_st.index, y=agg_st, name="Aggressive"),
        go.Bar(x=nonagg_st.index, y=nonagg_st, name="Not Aggressive"),
    ]
    fig = go.Figure(data=trace)
    fig.update_layout(title='Aggressive Language per Stance', barmode='group')
    st.plotly_chart(fig)
aggressive_by_stance(data)

st.write("I include a graph showing the frequency of aggressive and not aggressive tweets per stance to visualize the type of language each stance uses in discussing climate change.")