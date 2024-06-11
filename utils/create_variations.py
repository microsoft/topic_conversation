# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license. 

import os
import copy

import argparse
import json
import logging

import pandas as pd
import numpy as np

def define_fields_time():
    '''
    Define the time fields for meetings, topics, and transcripts.
    Returns:
        tuple: A tuple containing three lists: meeting_times, topic_times, trans_times.
    '''
    meeting_times = ['meeting_start_s', 'meeting_end_s']
    topic_times = ['topic_start_s', 'topic_end_s']
    trans_times = ['start_s', 'end_s']
    return meeting_times, topic_times, trans_times

def define_fields_line():
    '''
    Define the line fields for meetings, topics, and transcripts.
    Returns:
        tuple: A tuple containing three lists: meeting_lines, topic_lines, trans_lines.
    '''
    meeting_lines = ['meeting_start_line', 'meeting_end_line']
    topic_lines = ['topic_start_line', 'topic_end_line']
    trans_lines = ['line_id']
    return meeting_lines, topic_lines, trans_lines

def prepare_functions():
    '''
    Prepare and return a dictionary of variation functions.
    Returns:
        dict: A dictionary containing variation functions.
    '''
    var_functions = {
        'variation_addTopics': create_data_addTopics,  
        'variation_removeTopics': create_data_removeTopics, 
    }
    return var_functions

def prepare_kwargs():
    '''
    Prepare and return a dictionary of keyword arguments for variation functions.
    Returns:
        dict: A dictionary containing keyword arguments.
    '''
    kwargs = {
        'n_new_range': [1, 3],
        'n_remove_range': [1, 3],
        'n_min_topics': 2
    }
    return kwargs

def prepare_data(mv, meeting_times, topic_times, trans_times):
    '''
    Prepare the data by aligning all timestamps to start from 0 if specified in the arguments.
    Args:
        args (argparse.Namespace): The parsed command line arguments.
        mv (dict): The meeting data.
        meeting_times (list): The list of meeting time fields.
        topic_times (list): The list of topic time fields.
        trans_times (list): The list of transcript time fields.
    '''
    # Align all timestamps to start from 0
    mdf = pd.DataFrame.from_dict(mv['topics']).transpose()
    start_s = mdf['topic_start_s'].min()
    if start_s > 0:
        for tk, tv in mv['topics'].items():
            # Update topic level timestamps
            for ftime in topic_times:
                new_time = {ftime: tv[ftime]-start_s}
                tv.update(new_time)
            # Update transcripts level timestamps
            for ftime in trans_times:
                for i, trans in enumerate(tv['transcripts']):
                    new_time = {ftime: trans[ftime]-start_s}
                    tv['transcripts'][i].update(new_time)
        # Update meeting level
        for ftime in meeting_times:
            mv['metadata'][ftime] = mv['metadata'][ftime] - start_s

def create_variation_dist(n_input_meetings, variation_type, variation_ratio):
    '''
    Create a distribution of the number of variation meetings to be created based on the input parameters.
    Args:
        n_input_meetings (int): The number of input meetings.
        variation_type (str): The type of variation.
        variation_ratio (float): The ratio of variation meetings to be created.
    Returns:
        list: A list containing the number of variation meetings to be created for each input meeting.
    '''
    # If variation_ratio is 1.0 (100%), create one variation meeting from each input meeting
    if variation_ratio == 1.0:
        n_dist = [1]*n_input_meetings
    else:
        n_target_meetings = int(n_input_meetings * variation_ratio)
        # If variation_ratio < 1.0 (100%), select a subset to skip and create one variation meeting from each meeting that is left
        if variation_ratio < 1.0:
            n_omit = n_input_meetings - n_target_meetings
            idx_omit = np.random.choice(n_input_meetings, size=n_omit, replace=False)
            n_dist = [0 if i in idx_omit else 1 for i in range(n_input_meetings)]

        # If variation_ratio > 1.0 (100%), randomly create at least 1 variation meeting from each meeting. The total N may vary a bit from the exact target.
        else:
            lim_lower = max(1, int(variation_ratio*0.9))
            lim_upper = variation_ratio*1.1 + 1
            n_dist = np.random.randint(lim_lower, lim_upper+1, size=n_input_meetings)
    # Log
    print('{variation_type}: Try to create {n_to_create} meetings. Ratio: {ratio}'.format(
        variation_type=variation_type, n_to_create=sum(n_dist), ratio=sum(n_dist)/n_input_meetings))
    return n_dist

def prepare_data_all(args, data):
    '''
    Prepare the data for all meetings in the input data.
    Args:
        data (dict): The input data.
    '''
    meeting_times, topic_times, trans_times = define_fields_time()
    for sk, sv in data.items():
        for mi, (mk, mv) in enumerate(sv.items()):
            prepare_data(mv, meeting_times, topic_times, trans_times)
    prepare_output(args)

def create_data_addTopics(sv, dist_var, **kwargs):
    '''
    Create new meetings by adding topics to existing meetings.
    Args:
        sv (dict): The input meeting data.
        dist_var (list): The distribution of the number of variation meetings to be created.
        **kwargs: Additional keyword arguments.
    Returns:
        dict: The updated meeting data with added topics.
    '''
    variation_type = 'variation_addToics'
    sv_new = copy.deepcopy(sv)
    n_new_range = kwargs['n_new_range']
    for mi, (mk, mv) in enumerate(sv.items()):
        # Get topics from other meetings
        current_topics = list(set(mv['topics'].keys()))
        other_topics = list(set([tp for x in sv.keys() for tp in sv[x]['topics'].keys() if (x!=mk) and (tp not in current_topics)]))
        mnew = dist_var[mi]
        # Iterate based on number of to-be-created meetings
        list_vars = []
        if mnew>0:
            nadj = 0
            for inew in range(mnew):
                # New meeting
                i_name = '{0}_{1}_{2}'.format(mk, variation_type, inew-nadj)
                sv_new[i_name] = copy.deepcopy(mv)
                # Randomly pick topics from other meetings; skip if the sorted picks exist already
                i_add_n = min(np.random.randint(n_new_range[0], n_new_range[1]+1), len(other_topics))
                i_add_topics = list(np.random.choice(other_topics, size=i_add_n, replace=False))
                i_add_topics_sorted = sorted(i_add_topics)
                if i_add_topics_sorted in list_vars:
                    nadj = nadj + 1
                    continue
                list_vars.append(i_add_topics_sorted)
                # Add the new topics to data. 
                # Negative time and line information; empty transcripts information. 
                for iadd in i_add_topics:
                    iinfo = {
                        iadd: {
                            'topic_start_s': -1,
                            'topic_end_s': -1,
                            'topic_start_line': -1,
                            'topic_end_line': -1,
                            'topic_trans_word_count': 0,
                            'transcripts': [],
                        }

                    }
                    sv_new[i_name]['topics'].update(iinfo)
                # Metadata
                sv_new[i_name]['metadata'] = copy.deepcopy(sv[mk]['metadata'])
                sv_new[i_name]['metadata']['variations'].update({variation_type: i_add_topics})
        _ = sv_new.pop(mk)
    return sv_new

def create_data_removeTopics(sv, dist_var, **kwargs):
    '''
    Create new meetings by removing topics from existing meetings.
    Args:
        sv (dict): The input meeting data.
        dist_var (list): The distribution of the number of variation meetings to be created.
        **kwargs: Additional keyword arguments.
    Returns:
        dict: The updated meeting data with removed topics.
    '''
    # Set parameters
    variation_type = 'variation_removeTopics'
    sv_new = copy.deepcopy(sv)
    n_remove_range = kwargs['n_remove_range']
    n_min_topics = kwargs['n_min_topics']
    _, topic_times, trans_times = define_fields_time()
    _, topic_lines, trans_lines = define_fields_line()
    # Iterate through meetings
    for mi, (mk, mv) in enumerate(sv.items()):
        # Get topics from the current meeting. If the length is too short, skip.
        current_topics = list(mv['topics'].keys())
        if len(current_topics)<=n_min_topics:
            _ = sv_new.pop(mk)
            continue
        mnew = dist_var[mi]
        # Iterate based on number of to-be-created meetings
        list_vars = []
        if mnew>0:
            nadj = 0
            for inew in range(mnew):
                # New meeting
                i_name = '{0}_{1}_{2}'.format(mk, variation_type, inew-nadj)
                sv_new[i_name] = copy.deepcopy(mv)
                # Randomly pick topics to remove; skip if the sorted picks exist already or 0 was picked
                i_rm_max = len(current_topics) - n_min_topics
                i_rm_n = min(np.random.randint(n_remove_range[0], n_remove_range[1]+1), i_rm_max)
                i_rm_topics = list(np.random.choice(current_topics, size=i_rm_n, replace=False))
                i_rm_topics_sorted = sorted(i_rm_topics)
                if (i_rm_topics_sorted in list_vars) or (i_rm_n==0):
                    nadj = nadj + 1
                    continue
                list_vars.append(i_rm_topics_sorted)
                # Update information after each removal
                for irm in i_rm_topics:
                    # Remove the new topic from data
                    _ = sv_new[i_name]['topics'].pop(irm)
                    # Adjust time, line, word count information accordingly
                    irm_start_s = mv['topics'][irm]['topic_start_s']
                    irm_s_len = mv['topics'][irm]['topic_end_s'] - mv['topics'][irm]['topic_start_s']
                    irm_start_line = mv['topics'][irm]['topic_start_line']
                    irm_line_len = mv['topics'][irm]['topic_end_line'] - mv['topics'][irm]['topic_start_line'] + 1
                    irm_wc = mv['topics'][irm]['topic_trans_word_count']
                    for rk, rv in sv_new[i_name]['topics'].items():
                        if mv['topics'][rk]['topic_start_s']>irm_start_s:
                            # Topic level
                            for rvt in topic_times:
                                rv[rvt] = rv[rvt] - irm_s_len
                            for rvl in topic_lines:
                                rv[rvl] = rv[rvl] - irm_line_len
                            # Transcripts level
                            for itrm, tr in enumerate(rv['transcripts']):
                                for rvtt in trans_times:
                                    new_time = {rvtt: tr[rvtt] - irm_s_len}
                                    rv['transcripts'][itrm].update(new_time)
                                for rvtl in trans_lines:
                                    new_line = {rvtl: tr[rvtl] - irm_line_len}
                                    rv['transcripts'][itrm].update(new_line)
                # Metadata
                sv_new[i_name]['metadata'] = copy.deepcopy(sv[mk]['metadata'])
                new_metadata = {
                    'variations': {variation_type: i_rm_topics},
                    'meeting_start_s': np.min([x['topic_start_s'] for x in sv_new[i_name]['topics'].values()]),
                    'meeting_end_s': np.max([x['topic_end_s'] for x in sv_new[i_name]['topics'].values()]),
                    'meeting_start_line': np.min([x['topic_start_line'] for x in sv_new[i_name]['topics'].values()]),
                    'meeting_end_line': np.max([x['topic_end_line'] for x in sv_new[i_name]['topics'].values()]),
                    'meeting_trans_word_count': np.sum([x['topic_trans_word_count'] for x in sv_new[i_name]['topics'].values()])
                }
                sv_new[i_name]['metadata'].update(new_metadata)
        _ = sv_new.pop(mk)
    return sv_new

def read_data(args):
    '''
    Read data from a JSON file.
    Args:
        args (argparse.Namespace): Arguments containing the input data name.
    Returns:
        data (dict): The data read from the JSON file.
    '''
    input_data = 'data\{}.json'.format(args.input_data)
    with open(input_data, 'r') as jf:
        data = json.load(jf)
    return data

def augment_data(args, data):
    '''
    Augments the given data by creating new variations based on the specified parameters.
    Args:
        args (argparse.Namespace): The arguments object containing the variation parameters.
        data (dict): The input data dictionary.
    '''
    # Preparation
    prepare_data_all(args, data)
    var_functions = prepare_functions()
    kwargs= prepare_kwargs()
    # Iterate through data
    for sk, sv in data.items():
        n_input_meetings = len(sv)
        for variation_type in ['variation_addTopics', 'variation_removeTopics']:
            # New data info
            var_source_name = '{0}_{1}_{2}'.format(sk, args.new_name, variation_type.split('_')[-1])
            variation_value = getattr(args, variation_type)
            dist_var = create_variation_dist(n_input_meetings, variation_type, variation_value)
            # Create new daa
            sv_new = var_functions[variation_type](sv, dist_var, **kwargs)
            var_new_data = {var_source_name: sv_new}
            # Output data
            output_path = os.path.join(args.output_folder, '{0}_{1}.json'.format(var_source_name, len(sv_new)))
            with open(output_path, 'w') as nf:
                json.dump(var_new_data, nf, indent=4)
            print('{0} new meetings with {1} created. Final ratio: {2}\n'.format(len(sv_new), variation_type, len(sv_new)/n_input_meetings))

def prepare_output(args):
    '''
    Prepare the output folder for the augmented data.
    Args:
        args (argparse.Namespace): The parsed command line arguments.
    '''
    # Check if the output folder is an absolute path
    is_abs = os.path.isabs(args.output_folder)
    if not is_abs:
        args.output_folder = os.path.join(os.getcwd(), args.output_folder)
    # Check if the output folder exists; if not, create it
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)