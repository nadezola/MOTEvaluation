# MOT Robustness Evaluation

## Data pre-processing 
 
+ MOT17 dataset
  + MOT-17 GTs format: 
  `[fr_num, tr_id, x1, y1, w, h, conf, class, vis]`
  + classes: 
  `1: pedestrian, 2: cyclist, 3: car, 4: bicycle, 5: motorbike, 6: non motorized vehicle,
   7: static person, 8: distractor, 9: occluder, 10: occluder on the ground, 11: occluder full,
   12: reflection`

+ Evaluation format:
`[fr_num, tr_id, x1, y1, w, h, class=0, -1, -1, -1]`
  + Reformat and filter MOT17 GTs by class, visibility, etc.: `reformat_mot17_gts.py`.
  + Reformat tracking results to evaluation format: `reformat_res_tracks.py`. 

  
## Tracking Evaluation

+ Make sure, you have right mot17 labels in the folder: 
`data/mot17/train/mot_labels_quarter/*.txt` 
+ Make sure, you have tracking results in the right evaluation format in the folder: 
`data/restracks/TransCenter/mot17/train/<SPLIT>/tracks_for_eval/*.txt` 
+ Configure: `config/eval_config.py` 
+ Run evaluation:
  ```bash
  python evaluate.py --gts 'data/mot17/train/mot_labels_quarter'
                     --tracks 'data/restracks/TransCenter/mot17/train'  
                     --tracker_name 'TransCenter'
                     --out 'res_evaluation'
  ```
+ Run `make_summary.py` to put main evaluation results from previous step into a nice table
