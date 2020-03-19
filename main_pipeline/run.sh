#!/bin/bash
python 2a_post_process_for_tracking.py && python 2b_remove_overlap_boxes.py && python 3a_track_based_reid.py && python 3b_trajectory_processing.py && python 4a_match_tracks_for_crossroad.py && python 4b_match_tracks_for_arterialroad.py && python 5a_merge_results.py && python 5b_adapt_boxes.py && python 5c_convert_to_submission.py
