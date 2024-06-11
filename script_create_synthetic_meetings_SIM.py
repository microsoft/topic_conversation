# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license. 
# ========================================================================================================
# Description
#   This script is used to create synthetic meetings from the Speech Interruption Meeting (SIM) dataset. 
#   It takes command line arguments to specify the parameters for creating the synthetic dataset. 
#   The script generates synthetic meetings by randomly selecting sections from the original meetings and 
#   combining them to create new meetings. The generated synthetic meetings are then summarized and 
#   converted to JSON format.
#
# Arguments: 
#   --new_name: the name of the new dataset
#   --n_syn: the number of synthetic meetings to generate
#   --rdseed (optional): the random seed (default: 2)
#
# Example:
#   python script_create_synthetic_meetings_SIM.py --new_name syn100 --n_syn 100 --rdseed 2 
# ========================================================================================================

import argparse
import shutil
from utils.create_synthetic_SIM import *

def set_parameters(args):
    '''
    Set the parameters for creating synthetic dataset.
    Args:
        args (argparse.Namespace): Parsed command line arguments.
    Returns:
        syn_args (argparse.Namespace): Namespace object containing the set parameters.
    '''
    args_dict = {
        'new_dataset_name': args.new_name,
        'n_syn': args.n_syn,
        'remove_s_start': 5*60,  # Remove first 5 minutes of the original meeting to exclude setups
        'remove_s_end': 5*60, # Remove lst 5 minutes of the original meeting to exclude endings
        'max_len_s': 30*60, # Take maximum 30 minutes of a meeting
        'low_freq_topic_counts': 5, # Max number of occurrences to be considered a low frequency topic
        'n_topics_per_syn': [2, 5], # number of uniue topics per synthetic meeting (range)
        'leng_per_section_s': [5*60, 11*60] # length of each section in minutes (range)
        }
    syn_args = argparse.Namespace(**args_dict)
    return syn_args

def syn_parse_args():
    '''
    Parse command line arguments.
    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    '''
    parser = argparse.ArgumentParser(description='Parse inputs for creating synthetic dataset')
    parser.add_argument('--new_name', type=str, help='Name of the new dataset')
    parser.add_argument('--n_syn', type=int, help='Number of synthetic meetings')
    parser.add_argument('--rdseed', type=int, required=False, default=2, help='Random seed (default: 2)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    # Parameters
    args = syn_parse_args()
    syn_args = set_parameters(args)

    # Set random seed
    np.random.seed(args.rdseed)

    # Generate synthetic meetings
    ds_name, syn_info, tmp_path_out = generate_syn_meetings(syn_args)

    # Summarize synthetic meetings
    df_syn_info = pd.DataFrame(syn_info).transpose()
    print('Synthetic Meetings Summary:')
    print(df_syn_info[['n_sections', 'duration_total_m']].astype('float').describe())

    # Convert data to JSON format
    convert_data_to_json(ds_name, tmp_path_out, 'data')

    # Remove temporary files
    shutil.rmtree(tmp_path_out)