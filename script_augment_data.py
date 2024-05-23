# ========================================================================================================
# Description
#   This script is used to augment data by creating variations of the input data. 
#   The supported variations include adding random topics or removing topics and transcripts. 
#   The augmented data is saved in a separate file in the same format as the input data.
#
# Arguments: 
#     --input_data (str): Input data source name.
#     --new_name (str, optional): New data source name. Default is "New".
#     -a, --variation_addTopics (float, optional): Percentage of data to create by adding additional topics.
#                                                  Default is 0.5.
#     -r, --variation_removeTopics (float, optional): Percentage of data to create by removing topics and 
#                                                     transcripts. Default is 0.5.
#     --rdseed (int, optional): Random seed. Default is 2.
#     --output_folder (str, optional): The folder to save the augmented data. Default is "data".
#
# Example:
#     python script_augment_data.py --input_data ICSI --new_name aug -a 0.3 -r 0.3 --output_folder data/example_aug
# ========================================================================================================

import argparse
import numpy as np
from utils.create_variations import *

def aug_parse_args():
    '''
    Parse command line arguments.
    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_data', type=str, required=True, help='Input data source name.')
    parser.add_argument(
        '--new_name', type=str, required=False, default='New', help='New data source name. Default is "new".')
    parser.add_argument(
        '-a', '--variation_addTopics', type=float, required=False, default=0.5,
        help='Percentage of data to create by adding additional topics. 1.5 means 150% of the origianl data size. Default 0.5')
    parser.add_argument(
        '-r', '--variation_removeTopics', type=float, required=False, default=0.5,
        help='Percentage of data to create by removing topics and transcripts. 1.5 means 150% of the origianl data size. Default 0.5')
    parser.add_argument('--rdseed', type=int, required=False, default=2, help='Random seed (default: 2)')
    parser.add_argument('--output_folder', type=str, required=False, default='data', help='The folder to save the augmented data. Default is "data".')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Arguments
    args = aug_parse_args()

    # Set random seed
    np.random.seed(args.rdseed)

    # Read data
    data = read_data(args)

    # Generate new data
    augment_data(args, data)
