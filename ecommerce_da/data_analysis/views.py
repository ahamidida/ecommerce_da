from django.shortcuts import render
from .models import Order
import io
import urllib, base64

import pandas as pd
from matplotlib import pyplot as plt
import seaborn
import numpy as np
import scipy
from scipy.stats import skew
from scipy.stats import skewtest
from sklearn.cluster import KMeans
# Create your views here.


def index(request):
    plt.clf()
    data = Order.objects.all().values()
    df = pd.DataFrame.from_records(data)

    df=df.drop(df[df['total_purchase'] < 0].index)
    df=df[df['total_purchase']<= 761000.0]

    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    stats = df.groupby('weekday')['total_purchase'].agg(['mean', 'std'])
    weekday_table = pd.DataFrame({'Weekday': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                               'Mean': stats['mean'].values,
                               'Standard Deviation ': stats['std'].values,})
    
    df['day_type'] = df['weekday'].apply(lambda x: 'Weekend' if x in [3, 4] else 'Weekday')
    weekday_data = df.loc[df['day_type'] == 'Weekday', 'total_purchase']
    weekend_data = df.loc[df['day_type'] == 'Weekend', 'total_purchase']
    plt.hist(weekday_data, bins=20, alpha=0.5, color='blue', label='WorkingDays')
    plt.hist(weekend_data, bins=20, alpha=0.5, color='red', label='Weekends')
    plt.title('Daily Demand Distribution by Day Type')
    plt.xlabel('Total Purchase')
    plt.ylabel('Frequency')
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the PNG image in base64.
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    if request.method == 'POST':
        k_value = int(request.POST.get('K'))
    else:
        k_value=5
# df=pd.read_csv('sample_data.csv')
        # df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    latest_date = df['date'].max()

    rfm_scores = df.groupby('user_id').agg({
            'date' : lambda x: (latest_date - x.max()).days,
            'order_id': lambda x: len(x),
            'total_purchase':lambda x: x.sum()})

    rfm_scores['date'] = rfm_scores['date'].astype(int)

    rfm_scores.rename(columns={'date':'Recency',
                                'order_id':'Frequency',
                                'total_purchase':'Monetary'},inplace=True)
    quantiles = rfm_scores.quantile(q=[0.25,0.5,0.75])
    quantiles = quantiles.to_dict()
    def RScoring(x,p,d):
            if x <= d[p][0.25]:
                return 1
            elif x <= d[p][0.50]:
                return 2
            elif x <= d[p][0.75]: 
                return 3
            else:
                return 4
        
    def FnMScoring(x,p,d):
            if x <= d[p][0.25]:
                return 4
            elif x <= d[p][0.50]:
                return 3
            elif x <= d[p][0.75]: 
                return 2
            else:
                return 1

    def handle_neg_n_zero(num):
            if num <= 0:
                return 1
            else:
                return num
    rfm_scores['R'] = rfm_scores['Recency'].apply(RScoring, args=('Recency',quantiles,))
    rfm_scores['F'] = rfm_scores['Frequency'].apply(FnMScoring, args=('Frequency',quantiles,))
    rfm_scores['M'] = rfm_scores['Monetary'].apply(FnMScoring, args=('Monetary',quantiles,))
        #Apply handle_neg_n_zero function to Recency and Monetary columns 
    rfm_scores['Recency'] = [handle_neg_n_zero(x) for x in rfm_scores.Recency]
    rfm_scores['Monetary'] = [handle_neg_n_zero(x) for x in rfm_scores.Monetary]
    Log_Tfd_Data = rfm_scores[['Recency', 'Frequency', 'Monetary']].apply(np.log10, axis = 1).round(3)
    from sklearn.preprocessing import StandardScaler

        #Bring the data on same scale
    scaleobj = StandardScaler()
    Scaled_Data = scaleobj.fit_transform(Log_Tfd_Data)

        #Transform it back to dataframe
    Scaled_Data = pd.DataFrame(Scaled_Data, index = rfm_scores.index, columns = Log_Tfd_Data.columns)

    KMean_clust = KMeans(n_clusters= k_value, init= 'k-means++', max_iter= 10000)
    KMean_clust.fit(Scaled_Data)

        #Find the clusters for the observation given in the dataset
    rfm_scores['Cluster'] = KMean_clust.labels_

    plt.figure(figsize=(7,7))
    import random

    def generate_colors(num_clusters):
        # Generate a list of random RGB values
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_clusters)]
        # Convert RGB values to hex codes
        colors = ['#%02x%02x%02x' % c for c in colors]
        return colors
    Colors = generate_colors(k_value)
    rfm_scores['Color'] = rfm_scores['Cluster'].map(lambda p: Colors[p])
    ax = rfm_scores.plot(    
        kind="scatter", 
        x="Frequency", y="Recency",
        figsize=(10,8),
        c = rfm_scores['Color'],xlim=(0,60),ylim=(0,120)
    )

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the PNG image in base64.
    image_png = buffer.getvalue()
    buffer.close()
    graphic2 = base64.b64encode(image_png).decode('utf-8')

    avg_mon=rfm_scores.groupby('Cluster')['M'].agg(['mean'])
    avg_rec=rfm_scores.groupby('Cluster')['R'].agg(['mean'])
    avg_freq=rfm_scores.groupby('Cluster')['F'].agg(['mean'])
    reframed_cluster=rfm_scores['Cluster'].apply(lambda x: x + 1)
    rfm_table = pd.DataFrame({'Cluster': sorted(reframed_cluster.unique()),
                                'Ave. R': avg_rec['mean'].values,
                                'Ave. F' : avg_freq['mean'].values,
                                'Ave. M': avg_mon['mean'].values,})

    context={
        'table':weekday_table.to_html(),
        'plot1':graphic,
        'plot2':graphic2,
        'table2':rfm_table.to_html(),
    }
    
    return render(request,'data_analysis/index.html',context)

