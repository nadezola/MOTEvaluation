"""
    Puts main evaluation results in one nice table
"""
from pathlib import Path
import pandas as pd
import argparse
from config import eval_config


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--scene', default='clear', help='Choose the dataset split')
    parser.add_argument('--eval_root', default='res_evaluation/TransCenter',
                        help='Path to detailed evaluation results')
    parser.add_argument('--out_file', default='EVALUATION_SUMMARY.csv',
                        help='Output file (in the same folder with the detailed evaluation results')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    eval_root = Path(args.eval_root)
    out_filename = Path(args.out_file)

    eval_columns = ['Class', 'Seq', 'HOTA', 'MOTA', 'MOTP', 'IDF1', 'IDs', 'GT_IDs', 'ID_Sw', 'Dets', 'GT_Dets',
                    'Dets_TP', 'Dets_FN', 'Dets_FP']

    splits = eval_config.splits_opts['CONDITIONS']
    for split in splits:
        eval_path = eval_root / split
        eval_summary_file = eval_path / out_filename
        eval_file_list = sorted(list(eval_path.glob('*_detailed.csv')))
        eval_sum_df = pd.DataFrame(columns=eval_columns)
        for file in eval_file_list:
            class_name = file.stem.split('_')[0]
            class_eval_df = pd.read_csv(file)
            for i, row in class_eval_df.iterrows():
                if row['GT_IDs'] == 0:
                    continue

                eval_new_dict = {
                    'Class': [class_name + '_' + split],
                    'Seq': [row['seq']],
                    'HOTA': [round(row['HOTA(0)'] * 100, 2)],
                    'MOTA': [round(row['MOTA'] * 100, 2)],
                    'MOTP': [round(row['MOTP'] * 100, 2)],
                    'IDF1': [round(row['IDF1'] * 100, 2)],
                    'IDs': [row['IDs']],
                    'GT_IDs': [row['GT_IDs']],
                    'ID_Sw': [row['IDSW']],
                    'Dets': [row['Dets']],
                    'GT_Dets': [row['GT_Dets']],
                    'Dets_TP': [row['IDTP']],
                    'Dets_FN': [row['IDFN']],
                    'Dets_FP': [row['IDFP']]
                }
                eval_sum_df = pd.concat([eval_sum_df, pd.DataFrame(eval_new_dict)], ignore_index=True)

        eval_sum_df.to_csv(eval_summary_file)



