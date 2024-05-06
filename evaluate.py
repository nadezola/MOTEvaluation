import sys
import os
import argparse
from multiprocessing import freeze_support
from TrackEval import trackeval
from config import eval_config
from dataprocessing.dataset import TrackDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gts',  default='data/mot17/train/mot_labels_half',
                        help='Path to the folder with GTs (txt-files)')
    parser.add_argument('--tracks', default='data/restracks/ByteTrack/mot17/train',
                        help='Path to the folder with tracks in evaluation format (txt-files)')
    parser.add_argument('--tracker_name', default='ByteTrack', help='Tracker name for saving results')
    parser.add_argument('--quarter', action='store_true', help='Evaluate mot17-train-quarter dataset')
    parser.add_argument('--half', action='store_true', help='Evaluate mot17-train-half dataset')
    parser.add_argument('--out', action='store', default='res_evaluation', help='Path to save results')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    freeze_support()
    args = parse_args()

    eval_opts = eval_config.eval_opts
    metrics_opts = eval_config.metrics_opts
    splits = eval_config.splits_opts

    dataset_list = [TrackDataset(args, condition) for condition in splits['CONDITIONS']]

    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_opts['METRICS']:
            metrics_list.append(metric(metrics_opts))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')

    evaluator = trackeval.Evaluator(eval_opts)
    evaluator.evaluate(dataset_list, metrics_list)
