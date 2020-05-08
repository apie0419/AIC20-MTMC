## Preparation

Dateset download: [AICITY20-Track3-download](https://www.aicitychallenge.org/2020-data-and-evaluation/)

### Data we use

1. AICITY20-Track3
1. AICITY20-Track2

Note that the `det_reid_features.txt` is the middle result of `1b_merge_visual_feature_with_ other_feature.py`, and the other files are provided by organisers.

## Step by Step for MTMC Vehicle Tracking

### running code orderly

Modified the parameters described in the input session. (in config/config.yml)

#### 1_crop_vehicle_img_from_vdo.py

For each bounding box, crop the vehicle image and calculate the gps, according to the results of detection.

input:

- input_dir:`./aic20-track3-mtmc/train`or`./aic20-track3-mtmc/test`

output:

- for each video, produce `det_gps_feature.txt` to save gps information
- for each video, save all cropped image.

#### 1a_extract_visual_feature_for_each_img.py

extract reid feature for each corpped image, the train and inference pipeline follows reid-baseline

input:

- input_dir: `./aic20-track3-mtmc/train`
- pretrain_dir: `./weight/resnet50-19c8e357.pth`
- weight: `./weight/resnet50_model_120.pth`

ouptut:

- for each video, produce `deep_features.txt` file

**To train a new model for reid**

1. Prepare data.
2. Write your own dataset object in `./reid_baseline/data/datasets`. If you use aicity20-track3 dataset as your data. You need to run `process_gt_images.py` to cropped images and use `./reid_baseline/data/datasets/aic20_t3.py` as dataset object.
3. Write your configuration based on `./reid_baseline/configs/softmax_triple.yml`.
4. Following the instructions in `reid_baseline` folder.

#### 1b_merge_visual_feature_with_other_feature.py

Merge reid feature and gps information into one file.

input:

- input_dir: `./aic20-track3-mtmc/train`or`./aic20-track3-mtmc/test`
- gps information file `/det_gps_feature.txt`, from `1`
- ReID feature file `/deep_features.txt`, from `1a`

output:

- for each video, produce `det_reid_features.txt` file

#### 2_tracking.py

multi targets tracking for each video.

input:

- input_dir: `./aic20-track3-mtmc/train`or`./aic20-track3-mtmc/test`
- ID file `already_used_number.txt`, avoid reusing number
- `/det_reid_features.txt` from `1b`

output:

- for each video, produce tracking result file `det_reid_track.txt`

#### 2a_post_process_for_tracking.py

Optimize tracking result to solve target lost.

input:

- input_dir: `./aic20-track3-mtmc/train`or`./aic20-track3-mtmc/test`
- for each video, need `det_reid_track.txt` from `2`

output:

- for each video, produce tracking result `optimized_track.txt`

#### 2b_remove_overlap_boxes.py

Remove overlapped bounding box.

input:

- input_dir: `./aic20-track3-mtmc/train`or`./aic20-track3-mtmc/test`
- for each video, need `optimized_track.txt` from `2a`

output:

- for each video, produce tracking result `optimized_track_no_overlapped.txt`

#### 3_track_based_reid_deep.py

MTMC between all tracks.

input:

- input_dir: `./aic20-track3-mtmc/train`or`./aic20-track3-mtmc/test`
- for each video, need `optimized_track_no_overlapped.txt` from `2b`

output:

- ReID similarity file `submission`

**To train a new model for mtmc**
1. Prepare data.
2. Write your own dataset object in `./mt_tracker/data/datasets`. If you use aicity20-track3 dataset as your data. You need to run `process_gt_images.py` to cropped images, run `process_fps_file.py` to extract fps for every video, run `process_tracker_data.py` to extract tracks from ground truth file. 
3. use `./reid_baseline/data/datasets/aic20_t3.py` as dataset object.
4. Write your configuration based on `./reid_baseline/config/config.yml`.
5. run `train.py`

#### 4a_adapt_boxes.py

post process for each bounding box

input:

- input_dir: `./aic20-track3-mtmc/train`or`./aic20-track3-mtmc/test`
- `submission` from `3`

output:

- result file `submission_adpt`

#### 4b_convert_to_submission.py

convert the result to submission format

input:

- `submission_adpt` from `4a`

output:

- submission file `track3.txt`

### Guide for use

Run the code from `1_\*.py` to `4b_\*.py` orderly.
The train and inference for ReID follows reid-baseline
