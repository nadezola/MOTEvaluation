"""
Specify splits to process in config/eval_config.py
"""

from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from dataprocessing.utils import mkdir
from config.eval_config import splits_opts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks_root',
                        default='data/restracks/TransCenter/mot17/train',
                        help='Path to tracks folder')
    parser.add_argument('--tracker',
                        default='TransCenter',
                        choices=['FairMOT', 'ByteTrack', 'Tracktor++', 'TransCenter'],
                        help='Tracker name')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tracks_root = Path(args.tracks_root)
    tracker = args.tracker

    conditions = splits_opts['CONDITIONS']

    for condition in conditions:
        tracks_path = tracks_root / condition / 'tracks'
        out_path = tracks_root / condition / 'tracks_for_eval'
        mkdir(out_path)

        tracks_filelist = sorted(list(tracks_path.glob('*.txt')))
        for tracks_file in tqdm(tracks_filelist, desc=f'{condition}', total=len(tracks_filelist)):
            tracks = np.loadtxt(tracks_file, delimiter=',')

            if tracker == 'FairMOT':
                # FairMOT results:     [fr_num, tr_id, x1, y1, w, h, class=1, -1, -1, -1]
                # Evaluation format:   [fr_num, tr_id, x1, y1, w, h, class=0, -1, -1, -1]
                # => change class==1 to class=0
                tracks[tracks[:, 6] == 1, 6] = 0
            if tracker == 'ByteTrack':
                # ByteTracker results: [fr_num, tr_id, x1, y1, w, h, score, -1, -1, -1]
                # Evaluation format:   [fr_num, tr_id, x1, y1, w, h, class=0, -1, -1, -1]
                tracks = tracks[tracks[:, 6] > 0.6]
                tracks[:, 6] = 0
            if tracker == 'Tracktor++' or tracker == 'TransCenter':
                # Tracktor++ results:  [fr_num, tr_id, x1, y1, w, h, -1, -1, -1, -1]
                # Evaluation format:   [fr_num, tr_id, x1, y1, w, h, class=0, -1, -1, -1]
                tracks[:, 6] = 0

            tracks[:, 0] = tracks[:, 0] - tracks[:, 0].min() + 1
            tr_name = "-".join(tracks_file.stem.split("-")[0:2])
            np.savetxt(out_path / f'{tr_name}.txt', tracks.astype(int), fmt='%i', delimiter=',')



