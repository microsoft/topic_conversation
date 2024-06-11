# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license. 

import os
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

def syn_prepare(new_dataset_name):
    '''
    Prepare to create the synthetic dataset.
    Args:
        new_dataset_name (str): The name of the new dataset.
    Returns:
        tuple: A tuple containing the dataset name, input path, and output path.
    '''
    # Input
    path_in = r'data\SIM.json'
    # Output
    ds_name = 'SIM_{}'.format(new_dataset_name)
    tmp_path_out = os.path.join('data', 'tmp_{}'.format(datetime.now().strftime('%Y%m%d%H%M%S')))
    os.makedirs(tmp_path_out, exist_ok=False)
    return ds_name, path_in, tmp_path_out

def adjust_time(df_mdata):
    '''
    Adjusts the start and end times of a dataframe based on the minimum start time.
    Parameters:
        df_mdata (DataFrame): The input dataframe containing the start and end times.
    Returns:
        df_mdata (DataFrame): The modified dataframe with adjusted start and end times.
    '''
    min_start_s = df_mdata['start_s'].min()
    df_mdata['start_s'] = np.where(df_mdata['start_s']>=0, df_mdata['start_s']-min_start_s, df_mdata['start_s'])
    df_mdata['end_s'] = np.where(df_mdata['end_s']>=0, df_mdata['end_s']-min_start_s, df_mdata['end_s'])
    return df_mdata

def format_topics(df_mdata):
    '''
    Formats the topics in the given DataFrame by removing newlines and extra spaces.
    Args:
        df_mdata (DataFrame): The DataFrame containing the topics.
    Returns:
        df_mdata (DataFrame): The DataFrame with formatted topics.
    '''
    df_mdata['topic'] = df_mdata['topic'].apply(lambda x: ' '.join(x.replace('\n', ' ').replace('\r', ' ').split()))
    return df_mdata

def nested_defaultdict():
    """
    Create a nested defaultdict.
    Returns:
    - defaultdict: A nested defaultdict object.
    """
    return defaultdict(nested_defaultdict)
        
def convert_data_to_json(name_source, folder_source, folder_output):
    '''
    Converts data to JSON format and exports it to a file.
    Parameters:
        name_source (str): Name of the data source.
        folder_source (str): Path to the source folder.
        folder_output (str): Path to the output folder.
    '''

    # Get input files
    files_source = [os.path.join(folder_source, x) for x in list(os.walk(folder_source))[0][-1] if x.endswith('.csv') and x.startswith('transcript')]
    # Go over the files
    dict_source = nested_defaultdict()
    for f in tqdm(files_source, desc='Exporting {} transcripts...'.format(name_source)):
        # Get data
        try:
            tdf = pd.read_csv(f)
            fmeeting = '_'.join(os.path.split(f)[-1].split('_')[1:]).replace('.csv', '')
        except pd.errors.EmptyDataError:
            continue
        # Complete data
        tdf = convert_prep_complete_info(tdf, name_source)
        # Convert data to the target format
        for g, gdf in tdf.groupby('topic', sort=False):
            cols_keep = ['line_id', 'speaker', 'start_s', 'end_s', 'contents', 'word_count', 'cum_wc']
            cols_num = ['line_id', 'start_s', 'end_s', 'word_count', 'cum_wc']
            for c in cols_num:
                gdf[c] = gdf[c].astype(float)
            gtrans = gdf[cols_keep].to_dict(orient='records')
            gdict = {
                g: {
                    'topic_start_s': float(gdf['start_s'].min()),
                    'topic_end_s': float(gdf['end_s'].max()),
                    'topic_start_line': float(gdf['line_id'].min()),
                    'topic_end_line': float(gdf['line_id'].max()),
                    'topic_trans_word_count': float(gdf['word_count'].sum()),
                    'transcripts': gtrans
                }
            }
            # Add to the collection
            dict_source[name_source][fmeeting]['topics'].update(gdict)
        # Metadata
        dict_source[name_source][fmeeting]['metadata'] = {
            'topic_annotation_source': tdf['topic_source'].values[0],
            'timestamp_source': tdf['timestamp_source'].values[0], 
            'meeting_start_s': np.min([x['topic_start_s'] for x in dict_source[name_source][fmeeting]['topics'].values()]),
            'meeting_end_s': np.max([x['topic_end_s'] for x in dict_source[name_source][fmeeting]['topics'].values()]),
            'meeting_start_line': np.min([x['topic_start_line'] for x in dict_source[name_source][fmeeting]['topics'].values()]),
            'meeting_end_line': np.max([x['topic_end_line'] for x in dict_source[name_source][fmeeting]['topics'].values()]),
            'meeting_trans_word_count': np.sum([x['topic_trans_word_count'] for x in dict_source[name_source][fmeeting]['topics'].values()]),
            'variations': {}
        }
    # Output file
    output_source = os.path.join(folder_output, '{}.json'.format(name_source))
    with open(output_source, 'w') as osf:
        json.dump(dict_source, osf, indent=4) 
    print('Data saved to {}'.format(output_source))

def convert_prep_complete_info(tdf, name_source):
    '''
    Check and complete the necessary information before converting the data format
    Parameters:
        tdf (DataFrame): DataFrame containing the transcripts data
        name_source (str): Name of the data source
    Returns:
        tdf (DataFrame): Updated DataFrame with completed information
    '''
    # Sort
    if 'start_s' not in tdf.columns:
        tdf = estimate_time(tdf)
    else:
        tdf['timestamp_source'] = name_source
    tdf1 = tdf.loc[tdf['start_s']>=0].copy()
    tdf2 = tdf.loc[tdf['start_s']<0].copy()
    tdf = pd.concat([tdf1.sort_values('start_s', ascending=True), tdf2])
    # Complete information
    tdf['contents'].fillna('', inplace=True)
    if 'line_id' not in tdf.columns:
        tdf['line_id'] = tdf.index
    if 'word_count' not in tdf.columns:
        tdf['word_count'] = tdf['contents'].apply(lambda x: len([s for s in re.sub(r'[^\w\s]', '', x).split(' ') if s!='']))
    if 'cum_wc' not in tdf.columns:
        tdf['cum_wc'] = tdf['word_count'].cumsum()
    if 'topic_source' not in tdf.columns:
        tdf['topic_source'] = name_source
    return tdf

def estimate_time(df, words_per_minute=150):
    '''
    Estimate the start and end time of each transcript line based on the word count and words per minute.
    Parameters:
        df (DataFrame): DataFrame containing the transcripts data
        words_per_minute (int): Average number of words spoken per minute (default: 150)
    Returns:
        df (DataFrame): Updated DataFrame with start and end time columns
    '''
    if 'start_s' not in df.columns:
        # Prepare
        words_per_second = words_per_minute/60
        if 'word_count' not in df.columns:
            df['word_count'] = df['contents'].apply(lambda x: len([s for s in re.sub(r'[^\w\s]', '', x).split(' ') if s!='']))
        if 'cum_wc' not in df.columns:
            df['cum_wc'] = df['word_count'].cumsum()
        if 'cum_wc_start' not in df.columns:
            df['cum_wc_start'] = df['cum_wc'] - df['word_count']
        df['start_s'] = df['cum_wc_start'] / words_per_second
        df['end_s'] = df['cum_wc'] / words_per_second
        df['timestamp_source'] = 'estimated'
        df.drop(columns=['cum_wc_start'], inplace=True)
        return df
    else:
        return df
    
def generate_syn_meetings(syn_args):
    '''
    Generate synthetic meetings based on the given parameters.
    Args:
        syn_args (object): An object containing the parameters for generating synthetic meetings.
    Returns:
        tuple: A tuple containing the dataset name, synthetic meeting information, and the output path for the intermediate transcripts.
    '''
    # Prepare to create synthetic data
    ds_name, path_in, tmp_path_out = syn_prepare(syn_args.new_dataset_name)
    # Read SIM data
    dfs, df_topics = get_transcripts(path_in, syn_args.low_freq_topic_counts)
    # Generate Synthetic Transcripts
    syn_info = {}
    for i in tqdm(range(syn_args.n_syn), desc='Generating Synthetic Meetings'):
        # Randomly select topics, meetings, and content length based on the given parameters
        i_n_meetings, i_n_topics, i_l = syn_random_sel(df_topics, syn_args.n_topics_per_syn, syn_args.leng_per_section_s)
        # Generate synthetic meetings
        i_trans = pd.DataFrame()
        sction_start_s = 0
        for j, jm in enumerate(i_n_meetings):
            # Create synthetic meetings based on selected topics, meetings, and content length
            j_trans = syn_get_meeting_data(dfs[jm], syn_args.remove_s_start, syn_args.remove_s_end, syn_args.max_len_s)
            j_selected = syn_select_transcripts(j, j_trans, i_l)
            j_selected = syn_adjust_time(j_selected, sction_start_s)
            sction_start_s = j_selected['end_s'].max()
            # Format output
            j_selected = j_selected[['start_s', 'end_s', 'speaker', 'contents', 'topic', 'call_id', 'start_s_original', 'end_s_original']]
            i_trans = pd.concat([i_trans, j_selected], ignore_index=True)        
        # Format data
        i_name = syn_format_data(i, i_trans)
        # Save output
        syn_export_data(tmp_path_out, i_trans, i_name)
        # Add info
        syn_add_info(syn_info, i_name, i_trans, i_n_meetings, i_l)
    return ds_name, syn_info, tmp_path_out

def get_topics(data_topics, low_freq_topic_counts=5):
    '''
    Get topics from data_topics and assign a label based on their frequency. 
    If low frequency, assign 'Low_frequency_topics'; otherwise, keep the topic name.
    Parameters:
        data_topics (list): A list of topics.
        low_freq_topic_counts (int): The threshold for considering a topic as low frequency. Default is 5.
    Returns:
        df_topics (DataFrame): A DataFrame containing the topics, their frequencies, and the assigned labels.
    '''

    df_topics = pd.DataFrame(data_topics)
    df_topics['topic_freq'] = df_topics['topic'].map(df_topics['topic'].value_counts())
    df_topics['topic_sel'] = np.where(df_topics['topic_freq']>low_freq_topic_counts, df_topics['topic'], 'Low_frequency_topics')
    return df_topics

def get_transcripts(path_in, low_freq_topic_counts=5):
    '''
    Retrieve transcripts from a JSON file and process them.
    Args:
        path_in (str): The path to the JSON file containing the transcripts.
        low_freq_topic_counts (int): The threshold for considering a topic as low frequency.
    Returns:
        - dfs(dictionary): A dictionary of DataFrames, where each DataFrame represents the processed data for a meeting.
        - df_topics (DataFrame): A DataFrame containing information about the topics.
    '''
    dfs = {}
    data_topics = []
    with open(path_in, 'r') as f:
        trans_data = json.load(f)
    for vsource, v in trans_data.items():
        for m, mdata in tqdm(v.items(), desc='Processing {} data...'.format(vsource)):
            df_mdata = pd.DataFrame()
            for t, tdata in mdata['topics'].items():
                if tdata['transcripts'] == []:
                    tdata['transcripts'] = [{'contents': '.'}]
                df_tdata = pd.read_json('\n'.join([json.dumps(x) for x in tdata['transcripts']]), lines=True)
                df_tdata['topic'] = t
                df_mdata = pd.concat([df_mdata, df_tdata], ignore_index=True)
                data_topics.append({'call_id': m, 'topic': t})
            df_mdata.insert(0, 'call_id', m)
            df_mdata = adjust_time(df_mdata)
            df_mdata = format_topics(df_mdata)
            dfs[m] = df_mdata
    df_topics = get_topics(data_topics, low_freq_topic_counts)
    return dfs, df_topics

def syn_get_meeting_data(j_raw, remove_s_start, remove_s_end, max_len_s):
    '''
    Retrieves the meeting data from the raw data based on specified criteria.
    Parameters:
        j_raw (DataFrame): The raw data containing meeting information.
        remove_s_start (float): The first X seconds to remove from the meeting.
        remove_s_end (float): The last X seconds to remove from the meeting.
        max_len_s (float): The maximum length of the meeting in seconds.
    Returns:
        j_trans (DataFrame): The extracted meeting data that meets the criteria.
    '''
    keep_start = remove_s_start
    keep_end = min(j_raw['end_s'].max()-remove_s_end, keep_start+max_len_s)
    j_trans = j_raw.loc[(j_raw['start_s']>=keep_start)&(j_raw['end_s']<=keep_end)].copy().reset_index(drop=True)
    return j_trans

def syn_random_sel(df_topics, n_topics_per_syn=[2, 5], leng_per_section_s=[5, 11]):
    '''
    Randomly selects topics, meetings, and content lengths for synthetic data generation.
    Parameters:
        df_topics (DataFrame): DataFrame containing topic information.
        n_topics_per_syn (list): List specifying the range of number of meetings per synthetic data sample.
        leng_per_section_s (list): List specifying the range of content length per section in seconds.
    Returns:
        i_n_meetings (list): List of randomly selected meeting IDs.
        i_n_topics (ndarray): Array of randomly selected topic names.
        i_l (ndarray): Array of randomly selected content lengths in seconds.
    '''
    # Select topics
    i_n = np.random.randint(low=n_topics_per_syn[0], high=n_topics_per_syn[1]+1)
    i_n_topics = np.random.choice(df_topics['topic_sel'].unique(), size=i_n, replace=False)
    # Select meetings
    i_n_meetings = []
    for t in i_n_topics:
        t_meeting = np.random.choice(df_topics.loc[df_topics['topic_sel']==t, 'call_id'].values, size=1)[0]
        i_n_meetings.append(t_meeting)
    # Select content length
    i_l = np.random.uniform(low=leng_per_section_s[0], high=leng_per_section_s[1]+1, size=i_n)
    return i_n_meetings, i_n_topics, i_l

def syn_select_transcripts(j, j_trans, i_l):
    '''
    Selects transcripts based on specified criteria.
    Parameters:
        j (int): The index of the synthetic data.
        j_trans (DataFrame): The raw data containing meeting information.
        i_l (ndarray): Array of randomly selected content lengths in seconds.
    Returns:
        j_selected (DataFrame): The selected transcripts that meet the criteria.
    '''
    j_trans_end = j_trans['end_s'].max()
    j_trans_start_latest = j_trans_end - i_l[j]
    j_start = np.random.choice(j_trans.loc[j_trans['start_s']<j_trans_start_latest, 'start_s'], size=1)[0]
    j_end = j_start + i_l[j]
    j_selected = j_trans.loc[(j_trans['start_s']>=j_start)&(j_trans['end_s']<=j_end)].copy()
    return j_selected

def syn_adjust_time(j_selected, sction_start_s):
    '''
    Adjusts the start and end times of the selected transcripts based on the section start time.
    Parameters:
        j_selected (DataFrame): The selected transcripts.
        sction_start_s (float): The start time of the section.
    Returns:
        j_selected (DataFrame): The modified transcripts with adjusted start and end times.
    '''
    j_selected.rename(columns={'start_s': 'start_s_original', 'end_s': 'end_s_original'}, inplace=True)
    j_delta = j_selected['start_s_original'].min() - sction_start_s
    j_selected['start_s'] = j_selected['start_s_original'] - j_delta
    j_selected['end_s'] = j_selected['end_s_original'] - j_delta
    return j_selected

def syn_format_data(i, i_trans):
    '''
    Formats the synthetic data by adding the new meeting name and sorting the transcripts based on start time.
    Parameters:
        i (int): The index of the synthetic data.
        i_trans (DataFrame): The synthetic data containing the transcripts.
    Returns:
        i_name (str): The formatted name of the synthetic meeting.
    '''
    i_name = str(i).zfill(3)
    i_trans['call_id'] = 'SIMsyn_{}'.format(i_name)
    i_trans.sort_values('start_s', inplace=True)
    return i_name

def syn_export_data(tmp_path_out, i_trans, i_name):
    '''
    Exports the synthetic data to a CSV file.
    Parameters:
        i_trans (DataFrame): The synthetic data containing the transcripts.
        i_name (str): The name of the synthetic meeting.
    '''
    i_file_output = os.path.join(tmp_path_out, 'transcripts_SIMsyn_{}.csv'.format(i_name))
    i_trans.to_csv(i_file_output, index=False)

def syn_add_info(syn_info, i_name, i_trans, i_n_meetings, i_l):
    '''
    Retrieves the meeting data from the raw data based on specified criteria.
    Parameters:
        syn_info (dict): The dictionary to store the synthesized meeting information.
        i_name (str): The name of the synthesized meeting.
        i_trans (DataFrame): The synthesized meeting data containing the transcripts.
        i_n_meetings (list): The list of meeting IDs for the synthesized meeting.
        i_l (ndarray): The array of content lengths in seconds for the synthesized meeting.
    '''
    i_n = len(i_n_meetings)
    syn_info['SIMsyn_{}'.format(i_name)] = {
        'n_sections': i_n, 
        'duration_total_s': np.sum(i_l), 
        'duration_total_m': np.sum(i_l)/60, 
        'topic_list': list(i_trans['topic'].drop_duplicates().values), 
        'topic_duration_s': [str(round(x, 2)) for x in i_l],
        'meeting_list': i_n_meetings, 
    }