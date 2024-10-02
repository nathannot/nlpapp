import streamlit as st 
import pandas as pd
from gdeltdoc import GdeltDoc, Filters
from datetime import datetime, timedelta
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as sia
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_icon="🧊",
    layout="wide")
st.header('NLP Sentiment Analysis for Top Stocks')
st.write('Select from the following stocks. At the moment it only goes back three months.')



predefined_stock = st.selectbox(
    'Choose from the following popular stocks',
    ('AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META')
)

# Initialize GdeltDoc client
gd = GdeltDoc()

# Set the initial start and end dates
final_end_date = datetime.today().date()

if predefined_stock == 'AAPL':
    user = 'apple stock'
elif predefined_stock == 'MSFT':
    user = 'microsoft stock'
elif predefined_stock == 'AMZN':
    user = 'amazon stock'
elif predefined_stock == 'GOOGL':
    user = 'google stock'
elif predefined_stock == 'NVDA':
    user = 'nvidia stock'
elif predefined_stock == 'TSLA':
    user = 'tesla stock'
else:
    user = 'meta stock'


f = Filters(
    keyword = user,
    start_date = '2024-01-01',
    end_date = str(final_end_date),
    country = ["UK", "US"],
    num_records = 250
)

gd = GdeltDoc()

# Search for articles matching the filters
articlesb = gd.article_search(f)
b= articlesb[['seendate','title','language']]
b['seendate'] = pd.to_datetime(b.seendate)
b['date'] = b.seendate.dt.date
e = b.sort_values('date')
e = e[e.language == 'English']
start_date = final_end_date- timedelta(92)
apple = yf.download(predefined_stock, start =str(start_date), end=str(final_end_date))
analyser = sia()
e['sentiment'] = e['title'].apply(lambda x: analyser.polarity_scores(x)['compound'])
avg_sent = e.groupby('date')['sentiment'].mean().reset_index()
avg_sent.columns = ['Date', 'sentiment']
avg_sent.Date = pd.to_datetime(avg_sent.Date)
final = apple.reset_index().merge(avg_sent, how='left', on='Date').fillna(0)
final['polarity']=final['sentiment'].apply(lambda x: -1 if x<-0.1 else (1 if x>0.1 else 0))
final['label']=final['sentiment'].apply(lambda x: 'negative' if x<-0.1 else ('positive' if x>0.1 else 'neutral'))
fig = make_subplots(specs=[[{'secondary_y':True}]])
fig.add_trace(go.Scatter(x=final.Date, y=final.Close, name='price'), secondary_y=False)
fig.add_trace(go.Scatter(x=final.Date, y=final['sentiment'],
                         marker=dict(
        color= 'red'), name='sentiment'), secondary_y=True)
fig.update_yaxes(range=[-1,1],  secondary_y=True)  # Set a custom range for the secondary y-axis
fig.update_layout(hovermode='x')
st.subheader(f'NLP Analysis of {predefined_stock}')
col1, col2 = st.columns(2)
with col1:
    st.write('Price vs sentiment over last three months!')
    st.plotly_chart(fig)
with col2:
    fig1 = go.Figure()
    fig1.add_trace(go.Pie(labels= final.label.unique(),values=final.polarity.value_counts(),
                     hole = 0.4,
                     marker = dict(line=dict(color='black',width=1))))
    st.write('Pie Chart of Sentiment Breakdown')
    st.plotly_chart(fig1)

st.write('Downloadable Table of sentiment score for each day!')
st.write(avg_sent)
st.write('This are the kinds of NLP data that can be incoporated into machine learning models.')