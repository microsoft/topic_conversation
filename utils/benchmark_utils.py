import os
import glob
import json
import re

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report

def prep_folder(target_folder):
    '''
    Prepare the target folder by creating it if it does not exist.
    Parameters:
        target_folder (str): The folder path to prepare.
    '''
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

def locate_suffix_files(t_folder_input, suffix='*'):
    '''
    Find all file paths with the given suffix. Go through all subfolders within t_folder_input.
    Parameters:
        t_folder_input (str): The folder path to search for files.
        suffix (str): The file suffix to filter the files. Default is '*'.
    Returns:
        str: A list of file paths with the given suffix.
    '''
    # Generate a list of file paths with the given suffix in all subfolders
    t_pfiles = [y for x in os.walk(t_folder_input) for y in glob.glob(os.path.join(x[0], '*.{}'.format(suffix)))]
    return t_pfiles

def unify_quotes(str_prompt, style=None):
    '''
    Unify quotes in the given prompt based on the specified style, to avoid escaping characters being added to prompts.
    Parameters:
        str_prompt (str): The prompt string to unify quotes in.
        style (str): Optional. Specifies the style of quotes to unify. Valid values are no change (None), "single", "double", or an empty string. Default is None.
    Returns:
        str: The prompt string with unified quotes.
    Raises:
        ValueError: If an invalid style is provided.
    '''
    if style is not None:
        if style == 'single':
            str_prompt = str_prompt.replace('"', "'")
        elif style == 'double':
            str_prompt = str_prompt.replace("'", '"')
        elif style == '':
            str_prompt = str_prompt.replace('"', '').replace("'", '')
        else:
            raise ValueError('Invalid unify_quotes. Valid values are "single", "double", "", or None')
    return str_prompt

def prompt_topic_relevance(str_trans, str_topics, len_topics, quotes_style=''):
    '''
    Generate a prompt based on the template and inputs to determine the relevance of topics in a conversation transcript.
    Parameters:
        str_trans (str): The conversation transcript.
        str_topics (str): The list of topics being discussed.
        len_topics (int): The length of the topic list.
        quotes_style (str): Optional. Specifies the style of quotes to unify. Valid values are "single", "double", or an empty string. Default is an empty string.
    Returns:
        str: The generated prompt.
    '''
    prompt_template = '''HERE IS THE PROMPT TEMPATE THAT TAKES THE FOLLOWING INPUTS: {input_trans}, {input_topics}, {len_topics}. '''
    prompt_template = prompt_template.replace('\n    ', '\n')
    str_trans = unify_quotes(str_trans, style=quotes_style)
    str_prompt = prompt_template.format(input_trans=str_trans, input_topics=str_topics, len_topics=len_topics)
    return str_prompt

def write_prompts_to_json(list_prompt, p_folder_output):
    '''
    Write the list of prompts to a JSON file.
    Parameters:
        list_prompt (list): A list of prompts.
        p_folder_output (str): The output directory for the JSON file.
        p_type (str): The type of prompt.
    '''
    len_json = len(list_prompt)
    out_json_path = os.path.join(p_folder_output, 'prompts_tcr_{}.json'.format(len_json))
    with open(out_json_path, 'w') as f:
        f.write(pd.DataFrame(list_prompt).to_json(orient='records', lines=True))
    return out_json_path

def get_topics(df_mdata):
    '''
    Get the unique topics from the dataframe and return them as a list and a formatted string.
    Parameters:
        df_mdata (DataFrame): The dataframe containing the meeting data.
    Returns:
        tuple: A tuple containing the list of unique topics and the formatted string.
    '''
    topic_list = df_mdata['topic'].unique()
    topic_str = ', '.join([f"Topic{i+1}: {topic}" for i, topic in enumerate(topic_list)])
    return topic_list, topic_str

def get_topic_durations(df_c, topic_list):
    '''
    Calculate the duration of each topic in the meeting data and return a dictionary with the topic durations.
    Parameters:
        df_c (DataFrame): The dataframe containing the meeting data.
        topic_list (list): The list of topics.
    Returns:
        dict: A dictionary with the topic durations.
    '''
    df_c['duration'] = df_c['end_s'] - df_c['start_s']
    gt_topics = df_c.groupby('topic')['duration'].sum().sort_values(ascending=False).to_dict()
    for x in topic_list:
        gt_topics[x] = gt_topics.get(x, 0)
    gt_topics_ordered = {x: gt_topics[x] for x in topic_list}
    return gt_topics_ordered

def process_section(list_prompt, df_mdata, source, meeting, section_s=[300,600,900], min_n_words=300):
    '''
    Process each section of the meeting data and generate prompts based on the specified criteria.
    Parameters:
        list_prompt (list): The list to store the generated prompts.
        df_mdata (DataFrame): The dataframe containing the meeting data.
        source (str): The source of the meeting data.
        meeting (str): The meeting identifier.
        section_s (list): Optional. The list of section lengths in seconds. Default is [300, 600, 900].
        min_n_words (int): Optional. The minimum number of words required in a section. Default is 300.
    '''
    topic_list, topic_str = get_topics(df_mdata)
    for s in section_s:
        cuts = [x*s for x in range(int(df_mdata['end_s'].max()//s)+1)]
        for i, c in enumerate(cuts):
            if i>0:
                df_c = df_mdata.loc[df_mdata['start_s'].between(cuts[i-1], cuts[i])].copy()
                # Get contents
                str_trans = '. '.join(df_c.apply(lambda x: '<{0}> {1}'.format(x['speaker'], x['contents']), axis=1).to_list()) 
                # Skip a part if the contents are too short
                c_n_words = len([x for x in str_trans.split(' ') if not x.startswith('<')])
                if c_n_words < min_n_words:
                    continue
                # Get the topics' durations in the section
                gt_topics_ordered = get_topic_durations(df_c, topic_list)
                # Unique ID
                pid = "{isource}_{imeeting}_{isection}_{ii}".format(isource=source, imeeting=meeting, isection=s, ii=i)
                # Add info
                pdata = {
                    'source': source,
                    'meeting': meeting,
                    'section_length_s': s,
                    'start_s': df_c['start_s'].min(),
                    'end_s': df_c['end_s'].max(),
                    'promptId': pid,
                    'prompt': prompt_topic_relevance(str_trans, topic_str, len(topic_list), quotes_style=''), 
                    'gt_duration': gt_topics_ordered,
                }
                list_prompt.append(pdata)

def generate_relevance_prompts(t_pfiles):
    '''
    Generate prompts based on the meeting transcripts and topic relevance.
    Parameters:
        t_pfiles (list): A list of file paths to the meeting transcripts.
    Returns:
        list: A list of prompts.
    '''
    list_prompt = []
    for tf in t_pfiles:
        with open(tf, 'r') as jf:
            trans_data = json.load(jf)
        for vsource, v in trans_data.items():
            for m, mdata in v.items():
                df_mdata = pd.DataFrame()
                for t, tdata in mdata['topics'].items():
                    if tdata['transcripts'] == []:
                        tdata['transcripts'] = [{'contents': '.'}]
                    df_tdata = pd.read_json('\n'.join([json.dumps(x) for x in tdata['transcripts']]), lines=True)
                    df_tdata['topic'] = t
                    df_mdata = pd.concat([df_mdata, df_tdata], ignore_index=True)
                df_mdata.sort_values(['start_s'], ascending=True, inplace=True)
                process_section(list_prompt, df_mdata, vsource, m)
    return list_prompt


def parse_by_parts(vtext, response_parts):
    '''
    Locate the start and end indices of each response part in the given text.
    Args:
        vtext (str): The text to search for response parts.
        response_parts (dict): A dictionary containing the response parts as keys and their types as values.
    Returns:
        dict: A dictionary containing the start and end indices of each response part.
    '''
    # Locate start index
    pstarts = {}
    for p in response_parts.keys():
        try:
            pstarts[p]= vtext.index(p)
        except ValueError:
            continue
    # Locate end index and complete the list
    pidx = {}
    for p, pstart in pstarts.items():
        if pstart == max(pstarts.values()):
            pend = len(vtext) - 1
        else:
            pend = [x for x in pstarts.values() if x > pstart][0]
        pidx[p] = (pstart, pend)
    return pidx

def parse_list(in_str, len_target):
    '''
    Parse a list from the given string and validate its length and type.
    Args:
        in_str (str): The input string containing the list.
        len_target (int): The expected length of the list.
    Returns:
        list or None: The parsed list if it meets the length and type requirements, otherwise None.
    '''
    in_str = in_str.lower()
    try:
        # Find the first list index
        all_forward = [x.start() for x in re.finditer('\\[', in_str)]
        idx_back = in_str.index(']')
        try:
            idx_forward = [x for x in all_forward if x<idx_back][-1]
        except IndexError:
            return None
        # Get list
        klist = [x.strip() for x in (in_str[idx_forward+1:idx_back].replace('\n', ',')).split(',')]
        klist = [x.strip().capitalize() for x in klist]
        # Errors:
        # (1) length errors
        if len(klist) != len_target:
            return None
        # (2) type errors
        try:
            klist = [float(x.split(':')[-1].strip()) for x in klist]
        except:
            return None
    except ValueError:
        return None
    return klist


def parse_text(in_str):
    '''
    Format the input string and return the output string.
    Args:
        in_str (str): The input string to be formatted.
    Returns:
        str: The formatted output string.
    '''
    out_str = in_str
    return out_str

def get_gt_data(gt_path):
    '''
    Read the ground truth data from the specified file path.
    Args:
        gt_path (str): The path to the ground truth file.
    Returns:
        pd.DataFrame: The ground truth data as a pandas DataFrame.
    '''
    df_data = pd.read_json(gt_path, lines=True)
    all_gt = []
    for ir, row in df_data.iterrows():
        ir_id = row['promptId']
        ir_dict = row['gt_duration']
        n = 0
        for k, v in ir_dict.items():
            kl = k.strip()
            all_gt.append({'promptId': ir_id, 'n': n, 'topic': kl, 'gt_duration': v})
            n = n + 1
    df_gt = pd.DataFrame(all_gt)
    df_gt['gt_score'] = df_gt['gt_duration']/df_gt.groupby('promptId')['gt_duration'].transform('sum')
    df_gt['gt_score'] = np.where(df_gt['gt_duration']==0, 0, df_gt['gt_score']) # Fill na for certain meetings that none of the topics are discussed
    n_topics = df_gt.loc[df_gt['gt_duration']>0].groupby('promptId').size()
    df_gt['n_topics'] = df_gt['promptId'].map(n_topics)
    df_gt['1_topic'] = np.where(df_gt['n_topics']>1, 'multiple_topics', 'one_topic')
    return df_gt

def parse_responses(all_responses, df_gt, list_to_remove, response_parts):
    '''
    Parse the responses from the LLM completion and extract relevant information.
    Args:
        all_responses (dict): A dictionary containing the responses from LLM completion.
        df_gt (pd.DataFrame): The ground truth data as a pandas DataFrame.
        list_to_remove (list or str): The example(s) to be replaced in the responses.
        response_parts (dict): A dictionary containing the response parts as keys and their types as values.
    Returns:
        pd.DataFrame: The parsed responses as a pandas DataFrame.
        dict: A dictionary containing any error contents encountered during parsing.
    '''
    data_responses = []
    error_contents = {}
    col_score = ''.join(filter(str.isalnum, [k for k, v in response_parts.items() if v == 'list'][0]))
    for k, v in all_responses.items():
        if v is None:
            error_contents[k] = v
            continue
        kdata = []
        # Replace style strings
        if isinstance(list_to_remove, str):
            list_to_remove = [list_to_remove]
        for se in list_to_remove:
            vtext = v.replace(se, '')
        # Locate the start index of each reponse_parts
        kidx = parse_by_parts(vtext, response_parts)
        # Prepare to get data
        len_target = df_gt.loc[df_gt['promptId']==k].shape[0]
        kdata = {'promptId': k, 'n': [x for x in range(len_target)]}
        # Parse ieach part
        for p, ptype in response_parts.items():
            pname = ''.join(filter(str.isalnum, p))
            kpidx = kidx[p]
            kptext = vtext[kpidx[0]:kpidx[1]+1]
            if ptype == 'list':
                presult = parse_list(kptext, len_target)
                if presult is None:
                    error_contents[k] = v
                    continue
                else:
                    kdata[pname] = presult
            elif ptype == 'text':
                if k not in error_contents.keys():
                    kdata[pname] = parse_text(kptext)
            else:
                raise TypeError('The response part type can be only ["list", "text"]. Current value given: {}'.format(ptype))
        if k in error_contents.keys():
            continue
        data_responses.append(pd.DataFrame(kdata))
    # Combine all responses
    df_model = pd.concat(data_responses).reset_index(drop=True)
    df_model.rename(columns={col_score: 'model_score'}, inplace=True) # Keep the score as is. This can be modifed to accomodate different score aggregations
    # logs
    print('Parsed prompts: {}'.format(df_model['promptId'].nunique()))
    print('Parsed prompt*topic pairs: {}'.format(df_model.shape[0]))
    return df_model, error_contents

def prompt_styles():
    '''
    Define the response parts and style examples for parsing responses.
    Returns:
        dict: A dictionary containing the response parts as keys and their types as values.
        list or str: The style example(s) to be replaced in the responses.
    '''
    response_parts = {
        'THIS IS A RANDOM LIST': 'list',
    }
    list_to_remove = ['']
    return response_parts, list_to_remove

def bins_style(prompt_style='choices4'):
    '''
    Define the ground truth and model score buckets based on the prompt style.
    Args:
        prompt_style (str): The style of the prompt.
    Returns:
        dict: A dictionary containing the ground truth score buckets.
        dict: A dictionary containing the model score buckets.
    '''
    # Set binary threshold for ground truth: if a topic is discussed for less than X seconds, then say it's not discussed
    gt_s_not_discussed = 30
    # Settings for Adding Buckets
    gt_threshold_name = 'Threshold_{}s'.format(gt_s_not_discussed)
    gt_bins = {
        'buckets2': {
            gt_threshold_name: (-1, gt_s_not_discussed, 1e10)
        }
    }
    model_bins = {
        'choices4':{
            'buckets2': {
                'Threshold_0': (-1, 0.1, 3.1),
            },
        }
    }
    return gt_bins, model_bins[prompt_style]

def get_raw_completions(completion_path):
    '''
    Read the LLM completion JSON file and extract the responses for each prompt.
    Args:
        completion_path (str): The path to the LLM completion JSON file.
    Returns:
        dict: A dictionary containing the responses for each prompt.
    '''
    with open(completion_path, 'r') as llm_file:
        all_responses = json.load(llm_file)
    return all_responses

def join_results(df_gt, df_model, fname, output_dir):
    '''
    Merge the ground truth and model results, add score buckets, and export the parsed results.
    Args:
        df_gt (pd.DataFrame): The ground truth data as a pandas DataFrame.
        df_model (pd.DataFrame): The parsed responses from model as a pandas DataFrame.
        fname (str): The name of the file.
        output_dir (str): The output directory for the processed results.
    Returns:
        pd.DataFrame: The merged data with added score buckets as a pandas DataFrame.
        list: The list of bucket names.
    '''
    # Merge the results
    gt_bins, model_bins = bins_style()
    df_all_join = pd.merge(df_gt, df_model, on=['promptId', 'n'], how='right').sort_values(['promptId', 'n'])
    # Add buckets for ground truth
    for k, v in gt_bins.items():
        for tname, tbins in v.items():
            df_all_join['gt_{0}_{1}'.format(k, tname)] = pd.cut(df_all_join['gt_duration'], bins=tbins, right=True, labels=False).astype(int)
    # Add buckets for model results
    list_buckets = list(model_bins.keys())
    for k, v in model_bins.items():
        for tname, tbins in v.items():
            df_all_join['model_{0}_{1}'.format(k, tname)] = pd.cut(df_all_join['model_score'], bins=tbins, right=True, labels=False).astype(int)
    # Get threshold pairs to compare
    threshold_pairs = {}
    for k in list_buckets:
        threshold_pairs[k] = [(gt_key, model_key) for gt_key in gt_bins[k].keys() for model_key in model_bins[k].keys()]
    # Rearrange columns
    cols_gt = list(dict.fromkeys(['gt_duration', 'gt_score'] + [x for x in df_all_join.columns if x.startswith('gt_')]))
    cols_model = list(dict.fromkeys(['model_score'] + [x for x in df_all_join.columns if x.startswith('model_')]))
    cols_others = [x for x in df_all_join.columns if x not in cols_gt+cols_model]
    df_all_join = df_all_join[cols_others + cols_gt + cols_model]
    # Export
    df_all_join.to_csv(os.path.join(output_dir, '{}_parsed_results.csv'.format(fname)), index=False)
    return df_all_join, list_buckets, threshold_pairs

def calculate_metrics(df_all_join, bucket, metric_pair):
    '''
    Calculate various metrics for evaluating the performance of the model.
    Args:
        df_all_join (pd.DataFrame): The merged data with added score buckets as a pandas DataFrame.
        bucket (str): The name of the bucket.
        metric_pair (tuple): A tuple containing the names of the metrics to calculate correlations for.
    Returns:
        dict: A dictionary containing the calculated metrics for each topic and bucket.
        dict: A dictionary containing the calculated binary metrics for each topic and bucket.
    '''
    # Calculate correlations
    df_all = df_all_join.dropna().copy()
    metrics = defaultdict(lambda: defaultdict(dict))
    metrics_binary = defaultdict(lambda: defaultdict(dict))
    metrics['all']['correlation']['score'] = df_all[['gt_score', 'model_score']].corr().values[0,1]
    # Correlation
    # Note: correlation is less meaningful when -1 exists in the outputs
    metrics['all']['correlation'][bucket] = df_all[['gt_{0}_{1}'.format(bucket, metric_pair[0]), 'model_{0}_{1}'.format(bucket, metric_pair[1])]].corr().values[0,1]
    for g, dfg in df_all.groupby('1_topic'):
        metrics[g]['correlation']['score'] = dfg[['gt_score', 'model_score']].corr().values[0,1]
        metrics[g]['correlation'][bucket] = dfg[['gt_{0}_{1}'.format(bucket, metric_pair[0]), 'model_{0}_{1}'.format(bucket, metric_pair[1])]].corr().values[0,1]
    # Calculate accuracy
    metrics['all']['accuracy'][bucket] = (df_all['gt_{0}_{1}'.format(bucket, metric_pair[0])]==df_all['model_{0}_{1}'.format(bucket, metric_pair[1])]).astype(int).mean()
    for g, dfg in df_all.groupby('1_topic'):
        metrics[g]['accuracy'][bucket] = (dfg['gt_{0}_{1}'.format(bucket, metric_pair[0])]==dfg['model_{0}_{1}'.format(bucket, metric_pair[1])]).astype(int).mean()
    # Add precision, recall, FPR, F1 for binary classification
    if bucket=='buckets2':
        breports = classification_report(df_all['gt_{0}_{1}'.format(bucket, metric_pair[0])], df_all['model_{0}_{1}'.format(bucket, metric_pair[1])], digits=6, output_dict=True)
        breports['0']['FPR'] = 1 - breports['1']['recall']
        breports['1']['FPR'] = 1 - breports['0']['recall']
        metrics_binary['all']['Discussed'] = breports['1']
        metrics_binary['all']['NotDiscussed'] = breports['0']
        for g, dfg in df_all.groupby('1_topic'):
            greports = classification_report(dfg['gt_{0}_{1}'.format(bucket, metric_pair[0])], dfg['model_{0}_{1}'.format(bucket, metric_pair[1])], digits=6, output_dict=True)
            greports['0']['FPR'] = 1 - greports['1']['recall']
            greports['1']['FPR'] = 1 - greports['0']['recall']
            metrics_binary[g]['Discussed'] = greports['1']
            metrics_binary[g]['NotDiscussed'] = greports['0']
    return metrics, metrics_binary

def display_metrics(metrics, metrics_binary, title='', binary_only=True):
    '''
    Display the calculated metrics and binary metrics.
    Args:
        metrics (dict): A dictionary containing the calculated metrics for each topic and bucket.
        metrics_binary (dict): A dictionary containing the calculated binary metrics for each topic and bucket.
    '''
    df_metrics = pd.DataFrame.from_dict({
        (i,j): metrics[i][j] for i in metrics.keys() for j in metrics[i].keys()},
        orient='index')
    df_metrics.index.names = ['N_Topics', 'Metrics']
    df_metrics_binary = pd.DataFrame.from_dict({
        (i,j): metrics_binary[i][j] for i in metrics_binary.keys() for j in metrics_binary[i].keys()},
        orient='index').sort_index(level=-1, axis=0).sort_index(axis=1)
    df_metrics_binary.index.names = ['N_Topics', 'Target_Class']
    print(title)
    if not binary_only:
        print(df_metrics.to_string())
    print(df_metrics_binary.to_string())

