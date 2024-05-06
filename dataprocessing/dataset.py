from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from TrackEval.trackeval.datasets._base_dataset import _BaseDataset
from TrackEval.trackeval import _timing
from dataprocessing.utils import mkdir
from config import eval_config


class TrackDataset(_BaseDataset):
    """Dataset to evaluate in MOT Challenge format"""

    @staticmethod
    def get_default_dataset_config():
        ...


    def __init__(self, args, condition):
        super().__init__()

        self.quarter = args.quarter
        self.half = args.half
        self.condition = condition
        self.tracker_list = [args.tracker_name]
        self.config = {
            'GTs_ROOT': Path(args.gts),
            'TRACKS_ROOT': Path(args.tracks),
            'OUT_ROOT': Path(args.out) / self.tracker_list[0],
        }

        # Paths
        self.gts_fol = self.config['GTs_ROOT']
        self.tracks_fol = self.config['TRACKS_ROOT'] / self.condition / 'tracks_for_eval'
        self.output_fol = self.config['OUT_ROOT'] /self.condition
        mkdir(self.output_fol)

        # Change Default
        self.should_classes_combine = False  # Combine evaluation across all seqs in the class

        # Classes to eval
        self.classes_to_eval = {0: 'pedestrian'}
        self.class_list = list(self.classes_to_eval.values())

        # Sequences to eval
        self.seq_list = sorted([file.stem for file in self.tracks_fol.glob('*.txt')])
        if len(self.seq_list) < 1:
            raise Exception(f'No sequences are selected to be evaluated for split {self.condition}')

        # Sequences INFO
        if self.quarter:
            self.seq_info = eval_config.mot17_train_quarter
        elif self.half:
            self.seq_info = eval_config.mot17_train_half
        else:
            self.seq_info = eval_config.mot17_train


    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        if is_gt:
            file = self.gts_fol / f'{seq}.txt'
        else:
            file = self.tracks_fol / f'{seq}.txt'

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file, id_col=6, remove_negative_ids=True)

        # Convert data to required format
        num_timesteps = self.seq_info[seq][1] - self.seq_info[seq][0] + 1
        #num_timesteps = len(list((self.gts_fol.parent / 'images' / seq).glob('*')))
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys
        current_time_keys = [str(t + 1) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            if is_gt:
                text = 'Ground-truth'
            else:
                text = 'Tracking'
            raise Exception(
                text + ' data contains the following invalid timesteps in seq %s: ' % seq + ', '.join(
                    [str(x) + ', ' for x in extra_time_keys]))

        for t in range(num_timesteps):
            time_key = str(t + 1)
            if time_key in read_data.keys():
                try:
                    time_data = np.asarray(read_data[time_key], dtype=float)
                except ValueError:
                    if is_gt:
                        raise TrackEvalException(
                            'Cannot convert gt data for sequence %s to float. Is data corrupted?' % seq)
                    else:
                        raise TrackEvalException(
                            'Cannot convert tracking data from tracker %s, sequence %s to float. Is data corrupted?' % (
                                tracker, seq))
                try:
                    raw_data['dets'][t] = np.atleast_2d(time_data[:, 2:6])
                    raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                except IndexError:
                    if is_gt:
                        err = 'Cannot load gt data from sequence %s, because there is not enough ' \
                              'columns in the data.' % seq
                        raise TrackEvalException(err)
                    else:
                        err = 'Cannot load tracker data from tracker %s, sequence %s, because there is not enough ' \
                              'columns in the data.' % (tracker, seq)
                        raise TrackEvalException(err)
                if time_data.shape[1] >= 8:
                    raw_data['classes'][t] = np.atleast_1d(time_data[:, 6]).astype(int)
                else:
                    if not is_gt:
                        raw_data['classes'][t] = np.ones_like(raw_data['ids'][t])
                    else:
                        raise TrackEvalException(
                            'GT data is not in a valid format, there is not enough rows in seq %s, timestep %i.' % (
                                seq, t))
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.atleast_1d(time_data[:, 7].astype(int))}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, 7])
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
            if is_gt:
                raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    def class_name_to_class_id(self, cls):
        for key, val in self.classes_to_eval.items():
            if val == cls:
                return key
        return -1


    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        BDD100K:
            In BDD100K, the 4 preproc steps are as follow:
                1) There are eight classes (pedestrian, rider, car, bus, truck, train, motorcycle, bicycle)
                    which are evaluated separately.
                2) For BDD100K there is no removal of matched tracker dets.
                3) Crowd ignore regions are used to remove unmatched detections.
                4) All gt dets except pedestrian are removed, also removes pedestrian gt dets marked with zero_marked.
        """
        cls_id = self.class_name_to_class_id(cls)

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Only extract relevant dets for this class for preproc and eval (cls)
            gt_class_mask = np.atleast_1d(raw_data['gt_classes'][t] == cls_id)
            # gt_class_mask = gt_class_mask.astype(bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            # tracker_class_mask = tracker_class_mask.astype(np.bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            # Match tracker and gt dets (with hungarian algorithm)
            unmatched_indices = np.arange(tracker_ids.shape[0])
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_cols = match_cols[actually_matched_mask]
                unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)

            # For unmatched tracker dets, remove those that are greater than 50% within a crowd ignore region.
            unmatched_tracker_dets = tracker_dets[unmatched_indices, :]
            crowd_ignore_regions = raw_data['gt_crowd_ignore_regions'][t]
            intersection_with_ignore_region = self._calculate_box_ious(unmatched_tracker_dets, crowd_ignore_regions,
                                                                       box_format='x0y0x1y1', do_ioa=True)
            is_within_crowd_ignore_region = np.any(intersection_with_ignore_region > 0.5 + np.finfo('float').eps,
                                                   axis=1)

            # Apply preprocessing to remove unwanted tracker dets.
            to_remove_tracker = unmatched_indices[is_within_crowd_ignore_region]
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            data['gt_ids'][t] = gt_ids
            data['gt_dets'][t] = gt_dets
            data['similarity_scores'][t] = similarity_scores

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
        return similarity_scores

    def get_output_fol(self, tracker):
        return self.output_fol

    def get_display_name(self, tracker):
        return tracker + '-' + self.condition
