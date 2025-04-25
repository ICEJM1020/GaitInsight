""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2025-01-13
""" 
import argparse
import warnings
warnings.filterwarnings("ignore")

from config import CONFIG
from Processor.Processor import Processor
from Processor.tr_Processor import tr_Processor

parser = argparse.ArgumentParser(description='Action Recognition in Gait/TUG/S2S')

parser.add_argument('-a', '--action', 
                    choices=['train', 'evaluate'], 
                    default='train',
                    help='running type. \'train\' for training the model, \'evaluate\' for evaluating. (default: train)'
                    )
parser.add_argument('-m','--model',
                    choices=['', 'tr_'], 
                    default='tr_',
                    )
parser.add_argument('--datafile',
                    default='.temp/raw_data.txt',
                    help='raw bodyframe data file path.'
                    )


if __name__ == "__main__":
    args = parser.parse_args()
    CONFIG.load_config(f"configs/{args.model}{args.action}_gait.yaml")

    processor = eval(f"{args.model}Processor")()

    processor.start()