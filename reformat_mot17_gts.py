"""
Specify mot17-train-quarter dataset to process in config/eval_config.py
"""

from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from dataprocessing.utils import mkdir
from config import eval_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', action='store',
                        default='data/mot17/train/MOT17GTs',
                        help='Path to the folder with mot17-labels (txt-files)')
    parser.add_argument('--outname', action='store',
                        default='mot_labels_quater', help='Ouput path')
    parser.add_argument('--quarter', action='store_true',
                        help='Create a mot17-train-quarter labels file')
    parser.add_argument('--half', action='store_true',
                        help='Create a mot17-train-half labels file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = data_root.parent / args.outname
    mkdir(out_root)

    quarter = args.quarter
    half = args.half
    gt_files = sorted(list(data_root.glob('*.txt')))
    for gt_file in tqdm(gt_files):
        gts = np.loadtxt(gt_file, delimiter=',')

        # Sort by frame_num
        # gts = gts[gts[:, 0].argsort()]

        # Select classes = [1, 2]
        # gts = gts[(gts[:, 7] == 1)+gts[:, 7] == 2), :]

        # Select class = 1
        gts = gts[(gts[:, 7] == 1), :]

        # Select visibility >=0.3
        #gts = gts[gts[:, 8] >= 0.3, :]

        # Select frame numbers for half dataset
        if quarter:
            seq_name = "-".join(gt_file.stem.split("-")[0:2])
            start_frame = eval_config.mot17_train_quarter[seq_name][0]
            gts = gts[gts[:, 0] >= start_frame, :]
            gts[:, 0] = gts[:, 0] - start_frame + 1
        if half:
            seq_name = "-".join(gt_file.stem.split("-")[0:2])
            start_frame = eval_config.mot17_train_half[seq_name][0]
            gts = gts[gts[:, 0] >= start_frame, :]
            gts[:, 0] = gts[:, 0] - start_frame + 1

        # Class_label switch to position [6], "pedestrian" = 0
        gts[:, 6] = 0
        file_name = '-'.join(gt_file.stem.split("-")[0:2])
        np.savetxt(out_root / f'{file_name}.txt', gts.astype(int), fmt='%i', delimiter=',')

    print(f'Filtered mot labels saved in "{out_root}".')


