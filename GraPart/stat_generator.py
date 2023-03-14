import json 
import pandas as pd
import os
import sys
import plotly.graph_objects as go
import time

SAVE_PATH = 'Benchmark_results/'

def read_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

def read_MO_data(folder_name):
    df = pd.DataFrame()
    all_data = []
    for file in os.listdir(folder_name):
        if file.endswith('.json'):
            data = read_json(f'{folder_name}/' + file)
            all_data += data

    df = pd.json_normalize(all_data)
    df = df.groupby('number_of_clusters', as_index=False).mean()
    return df

def generate_graphs_for_MO(save_dir):

    read_dir = f'{SAVE_PATH}{save_dir}/raw_results'
    
    if not os.path.exists(f'{SAVE_PATH}{save_dir}/graphs'):
        os.makedirs(f'{SAVE_PATH}{save_dir}/graphs')
    


    df = read_MO_data(read_dir)

    # Create a sub-dataframe with just the columns we want to use
    df_firewall_self = df[['number_of_clusters','KMeans.firewalls_self', 'multiway.firewalls_self', 'oneway.firewalls_self']]

    # Rename columns
    df_firewall_self.columns = ['number_of_clusters','KMeans', 'multiway', 'oneway']

    # Create a bar plot for each 'number_of_clusters'
    fig = go.Figure(data=[
        go.Bar(name='KMeans', x=df_firewall_self['number_of_clusters'], y=df_firewall_self['KMeans']),
        go.Bar(name='Multiway', x=df_firewall_self['number_of_clusters'], y=df_firewall_self['multiway']),
        go.Bar(name='Oneway', x=df_firewall_self['number_of_clusters'], y=df_firewall_self['oneway'])
    ])

    # Update layout properties
    fig.update_layout(title='Number of firewalls (self variation)', xaxis_title='number_of_clusters', yaxis_title='Number of firewalls')
    fig.update_layout(barmode='group')

    #save as img
    fig.write_image(f'{SAVE_PATH}{save_dir}/graphs/self_variation_MO.png',width=1000, height=500, scale=2)
    fig.write_html(f'{SAVE_PATH}{save_dir}/graphs/self_variation_MO.html')



    df_firewall_other = df[['number_of_clusters','KMeans.firewalls_other', 'multiway.firewalls_other', 'oneway.firewalls_other']]
    df_firewall_other.columns = ['number_of_clusters','KMeans', 'multiway', 'oneway']

    fig = go.Figure(data=[
        go.Bar(name='KMeans', x=df_firewall_other['number_of_clusters'], y=df_firewall_other['KMeans']),
        go.Bar(name='Multiway', x=df_firewall_other['number_of_clusters'], y=df_firewall_other['multiway']),
        go.Bar(name='Oneway', x=df_firewall_other['number_of_clusters'], y=df_firewall_other['oneway'])
    ])

    fig.update_layout(title='Number of firewalls (other variation)', xaxis_title='number_of_clusters', yaxis_title='Number of firewalls')
    fig.update_layout(barmode='group')
    fig.write_image(f'{SAVE_PATH}{save_dir}/graphs/other_variation_MO.png',width=1000, height=500, scale=2)
    fig.write_html(f'{SAVE_PATH}{save_dir}/graphs/other_variation_MO.html')

    df_time = df[['number_of_clusters','multiway.time', 'oneway.time']]
    df_time.columns = ['number_of_clusters','multiway', 'oneway']

    fig = go.Figure(data=[
        go.Bar(name='Multiway', x=df_time['number_of_clusters'], y=df_time['multiway']),
        go.Bar(name='Oneway', x=df_time['number_of_clusters'], y=df_time['oneway'])
    ])

    fig.update_layout(title='Execution time (ms) ', xaxis_title='number_of_clusters', yaxis_title='Number of firewalls')
    fig.update_layout(barmode='group')
    fig.write_image(f'{SAVE_PATH}{save_dir}/graphs/time(ms)_MO.png',width=1000, height=500, scale=2)
    fig.write_html(f'{SAVE_PATH}{save_dir}/graphs/time(ms)_MO.html')

    df_convergence = df[['number_of_clusters','multiway.count', 'oneway.converge']]
    df_convergence.columns = ['number_of_clusters','multiway', 'oneway']
    fig = go.Figure(data=[
        go.Bar(name='Multiway', x=df_convergence['number_of_clusters'], y=df_convergence['multiway']),
        go.Bar(name='Oneway', x=df_convergence['number_of_clusters'], y=df_convergence['oneway'])
    ])
    fig.update_layout(title='Number of iteration to converge', xaxis_title='number_of_clusters', yaxis_title='Number of firewalls')
    fig.update_layout(barmode='group')
    fig.write_image(f'{SAVE_PATH}{save_dir}/graphs/converge_MO.png',width=1000, height=500, scale=2)
    fig.write_html(f'{SAVE_PATH}{save_dir}/graphs/converge_MO.html')

#---------------------------- BISECTION GRAPH GENERATION ----------------------------
def read_Bisection_data(folder_name):
    all_data = []
    for file in os.listdir(folder_name):
        if file.endswith('.json'):
            data = read_json(f'{folder_name}/' + file)
            all_data += data

    df = pd.DataFrame(all_data)
    df['clusters'] = df['clusters'].apply(lambda x: None if x=='max_firewalls not enough' else x)
    df = df.dropna()
    df = df.groupby('max_firewalls', as_index=False).mean()
    return df

def generate_graphs_for_Bisection(save_dir):

    read_dir = f'{SAVE_PATH}{save_dir}/raw_results'
    
    if not os.path.exists(f'{SAVE_PATH}{save_dir}/graphs'):
        os.makedirs(f'{SAVE_PATH}{save_dir}/graphs')

    df = read_Bisection_data(read_dir)
    df_firewall_self = df[['max_firewalls', 'clusters']]
    df_firewall_self.columns = ['max_firewalls', 'clusters']

    fig = go.Figure(data=[
        go.Bar(name='Clusters', x=df_firewall_self['max_firewalls'], y=df_firewall_self['clusters'])
    ])
    fig.update_layout(title='Number of clusters given max avaliable firwalls', xaxis_title='Max firewalls', yaxis_title='Number of clusters')
    fig.update_layout(barmode='group')
    fig.write_image(f'{SAVE_PATH}{save_dir}/graphs/clusters_given_firewalls.png',width=1000, height=500, scale=2)
    fig.write_html(f'{SAVE_PATH}{save_dir}/graphs/clusters_given_firewalls)_MO.html')




if __name__ == '__main__':
    generate_graphs_for_Bisection('bisection_2') 