"""
MOT17 GTs:
    [fr_num, tr_id, x1, y1, w, h, conf, class, vis]
classes:
    1: pedestrian, 2: cyclist, 3: car, 4: bicycle, 5: motorbike, 6: non motorized vehicle,
    7: static person, 8: distractor, 9: occluder, 10: occluder on the ground, 11: occluder full,
    12: reflection
"""

mot17_train = {
    'MOT17-02': [1, 600],     # [start_frame, end_frame]
    'MOT17-04': [1, 1050],
    'MOT17-05': [1, 837],
    'MOT17-09': [1, 525],
    'MOT17-10': [1, 654],
    'MOT17-11': [1, 900],
    'MOT17-13': [1, 750],
}

mot17_train_quarter = {
    'MOT17-02': [451, 600],     # [start_frame, end_frame]
    'MOT17-04': [789, 1050],
    'MOT17-05': [629, 837],
    'MOT17-09': [395, 525],
    'MOT17-10': [492, 654],
    'MOT17-11': [676, 900],
    'MOT17-13': [564, 750],
}

mot17_train_half = {
    'MOT17-02': [302, 600],     # [start_frame, end_frame]
    'MOT17-04': [527, 1050],
    'MOT17-05': [420, 837],
    'MOT17-09': [264, 525],
    'MOT17-10': [329, 654],
    'MOT17-11': [452, 900],
    'MOT17-13': [377, 750],
}

splits_opts = {
    'CONDITIONS': [
        'clear',
        'hetero_0.5/fog1',
        'hetero_0.5/fog2',
        'hetero_0.5/fog3',
        'hetero_0.5/fog4',
        'homo/fog1',
        'homo/fog2',
        'homo/fog3',
        'homo/fog4'
    ],
}

metrics_opts = {
    'METRICS': ['CLEAR', 'HOTA', 'Identity']    # Valid ['CLEAR', 'HOTA', 'Identity']
}

eval_opts = {
    'USE_PARALLEL': False,
    'NUM_PARALLEL_CORES': 8,
    'BREAK_ON_ERROR': True,  # Raises exception and exits with error
    'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
    'LOG_ON_ERROR': 'error_log.txt',  # if not None, save any errors into a log file.

    'PRINT_RESULTS': True,
    'PRINT_ONLY_COMBINED': False,
    'PRINT_CONFIG': True,
    'TIME_PROGRESS': True,
    'DISPLAY_LESS_PROGRESS': False,

    'OUTPUT_SUMMARY': True,
    'OUTPUT_EMPTY_CLASSES': False,  # If False, summary files are not output for classes with no detections
    'OUTPUT_DETAILED': True,
    'PLOT_CURVES': True,
}