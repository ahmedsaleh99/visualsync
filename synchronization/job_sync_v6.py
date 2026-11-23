import os
import glob
import cv2
import time
import numpy as np
import pickle
from itertools import combinations
from tqdm import tqdm
import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from collections import defaultdict


def group_folders_by_prefix(folders):
    groups = defaultdict(list)
    for folder in folders:
        # Split by underscore and take the first token as the group key
        prefix = folder.split("/")[-1].split('_')[0]
        groups[prefix].append(folder)
    return groups


def is_pickle_valid(filepath):
    try:
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
            if "offset_error_list" in result:
                if not (np.isnan(result["offset_error_list"]).all()==True):
                    return True
            return True
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--dataset_root", type=str, default="/data11/shaowei3/datasets/egohumans/data2_preprocessed_corrected", help="input dataset root")
    parser.add_argument("--result_root", type=str, default="/data11/shaowei3/datasets/egohumans/data2_preprocessed_corrected_results_v2_improved", help="result root")
    parser.add_argument('--offset_range', default=51, type=int)
    """
    blender: --offset_range 30
    panoptic: --offset_range 100
    egohumans: --offset_range 51
    """

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)

    # addtional arguments
    parser.add_argument('--moving_threshold', default=1, type=float, help="moving pixel filter threshold")
    parser.add_argument('--pixel_threshold', default=4, type=float, help="poisson disk sampling threshold")
    parser.add_argument("--max_batch_size", type=int, default=1024, help="max batch size for filtering correspondences")
    parser.add_argument("--max_N", type=int, default=30000, help="max number of correspondences")

    parser.add_argument("--skip_exist", action="store_true", help="skip existing results")

    parser.add_argument("--enable_poisson", action='store_true', help="enable poisson disk sampling")
    parser.add_argument("--use_weight", action='store_true', help="use weighted error")
    parser.add_argument("--not_use_v2", action='store_true', help="use v1 result")

    parser.add_argument("--use_vggt", action='store_true', help="use vggt for camera pose")
    parser.add_argument("--vggt_choice", default="", type=str, choices=["", "samplev1", "samplev2", "full"], help="vgg sample choice")
    parser.add_argument("--use_hloc", action='store_true', help="use hloc camera pose")
    parser.add_argument("--use_gt_intrinsic", action='store_true', help="use gt intrinsic") # default is false
    parser.add_argument("--use_gt_sample", action='store_true', help="use gt sample")

    parser.add_argument("--use_F_list", action='store_true', help="use F_list")
    parser.add_argument("--group_name", type=str, default=None, help="group name")
    parser.add_argument("--use_epipolar", action='store_true', help="use epipolar distance")
    parser.add_argument("--disable_gt", action='store_true', help="disable gt")  # used for in-the-wild video
    parser.add_argument("--use_cosine", action='store_true', help="use cosine distance")  # used for in-the-wild video
    parser.add_argument("--use_algebraic", action='store_true', help="use algebraic distance")

    args = parser.parse_args()

    script = Path(__file__).with_name("shaowei_sync_v6.py").resolve()
    os.makedirs(args.result_root, exist_ok=True)

    log_file = os.path.join(args.result_root, "sync_v6.log")

    video_dirs = sorted(glob.glob(os.path.join(args.dataset_root, "*/")))
    video_dirs = [video.rsplit('/', 1)[0] for video in video_dirs if os.path.isdir(video) and '*' not in video]

    groups = group_folders_by_prefix(video_dirs)

    pairs = []
    for prefix, folders in groups.items():
        if args.group_name is not None and prefix != args.group_name:
            continue
        print("prefix", prefix, "num_folders", len(folders))
        folders.sort()
        pairs_subgroup = list(combinations(folders, 2))
        pairs.extend(pairs_subgroup)

    print(f"total {len(pairs)} pairs")

    if args.skip_exist:
        valid_pairs = []
        for video1, video2 in tqdm(pairs):
            video1_name = os.path.basename(video1)
            video2_name = os.path.basename(video2)

            prefix = video1_name.split("_")[0]
            result_dir = os.path.join(args.result_root, prefix, video1_name + "__" + video2_name)
            if not args.not_use_v2:
                if args.use_vggt:
                    if args.vggt_choice != "":
                        result_path = os.path.join(result_dir, "result_candidates_v3_v2_vggt_{}.pkl".format(args.vggt_choice))
                    else:
                        result_path = os.path.join(result_dir, "result_candidates_v3_v2_vggt.pkl")
                elif args.use_hloc:
                    result_path = os.path.join(result_dir, "result_candidates_v3_v2_hloc.pkl")
                elif args.use_gt_sample:
                    result_path = os.path.join(result_dir, "result_candidates_v3_v2_gt_sample.pkl")
                elif args.use_F_list:
                    result_path = os.path.join(result_dir, "result_candidates_v3_v2_F_list.pkl")
                else:
                    result_path = os.path.join(result_dir, "result_candidates_v3_v2.pkl")
                if args.use_epipolar:
                    result_path = result_path.replace(".pkl", "_epipolar.pkl")
                if args.use_cosine:
                    result_path = result_path.replace(".pkl", "_cosine.pkl")
                if args.use_algebraic:
                    result_path = result_path.replace(".pkl", "_algebraic.pkl")
                if args.use_gt_intrinsic:
                        result_path = result_path.replace(".pkl", "_gt_intrinsic.pkl")
            else:
                result_path = os.path.join(result_dir, "result_candidates_v3.pkl")

            if os.path.exists(result_path) and is_pickle_valid(result_path):
                print(f"result {result_path} exists, skip")
                continue
            else:
                valid_pairs.append((video1, video2))

        print(f"total {len(valid_pairs)} unexist pairs")
        pairs = valid_pairs

    if args.end == -1:
        args.end = len(pairs) -1

    pairs = pairs[args.start: args.end + 1]

    runtime = []
    failures = 0
    matches_name = "tracks_match.npz" if args.not_use_v2 else "tracks_match_v2.npz"
    for video1, video2 in tqdm(pairs):
        video1_name = os.path.basename(video1)
        video2_name = os.path.basename(video2)

        prefix = video1_name.split("_", 1)[0]
        pair_dir = Path(args.result_root) / prefix / f"{video1_name}__{video2_name}"
        if not (pair_dir / matches_name).is_file():
            print(f"Skipping {video1_name} and {video2_name}: no precomputed correspondences")
            continue

        command = [sys.executable, str(script)]
        options = {
            "dataset_root": args.dataset_root,
            "result_root": args.result_root,
            "video1_name": video1_name,
            "video2_name": video2_name,
            "moving_threshold": args.moving_threshold,
            "pixel_threshold": args.pixel_threshold,
            "max_batch_size": args.max_batch_size,
            "offset_range": args.offset_range,
            "max_N": args.max_N,
            "vggt_choice": args.vggt_choice,
        }
        for name, value in options.items():
            command.extend([f"--{name}", str(value)])
        for flag in (
            "enable_poisson", "use_weight", "use_vggt", "use_hloc",
            "use_gt_intrinsic", "use_gt_sample", "use_F_list",
            "use_epipolar", "disable_gt", "use_cosine", "use_algebraic",
        ):
            if getattr(args, flag):
                command.append(f"--{flag}")
        if not args.not_use_v2:
            command.append("--use_v2")

        start_time = time.time()
        try:
            subprocess.run(
                command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True,
            )
        except subprocess.CalledProcessError as err:
            failures += 1
            with open(log_file, "a") as log:
                log.write(f"{video1_name} / {video2_name}\n{err.stdout}\n")
            print(f"Failed {video1_name} and {video2_name}. See {log_file}")
            continue
        end_time = time.time()
        elapsed_time = end_time - start_time
        runtime.append(elapsed_time)
        print(f"Processed {video1_name} and {video2_name} in {elapsed_time:.2f} seconds")
    if len(runtime) > 0:
        avg_runtime = sum(runtime) / len(runtime)
        print(f"Average runtime: {avg_runtime:.2f} seconds")


    if failures:
        raise SystemExit(f"{failures} pair scoring jobs failed")
