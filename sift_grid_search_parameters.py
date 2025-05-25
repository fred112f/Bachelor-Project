import os
import cv2
import numpy as np
import random
import itertools
import csv
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from threading import Lock
import sys

CACHE = {}
lock = Lock()

def build_cache(converted_folder, satellite_folder, file_list, base_sift):
    cache = {}
    for fn in file_list:
        conv_path = os.path.join(converted_folder, fn)
        sat_path = os.path.join(satellite_folder, fn)
        img_conv = cv2.imread(conv_path, cv2.IMREAD_GRAYSCALE)
        img_sat = cv2.imread(sat_path, cv2.IMREAD_GRAYSCALE)
        if img_conv is None or img_sat is None:
            continue
        kp_conv, desc_conv = base_sift.detectAndCompute(img_conv, None)
        kp_sat, desc_sat = base_sift.detectAndCompute(img_sat, None)
        if desc_conv is None or desc_sat is None:
            continue
        coords_conv = np.array([kp.pt for kp in kp_conv], dtype=np.float32)
        coords_sat = np.array([kp.pt for kp in kp_sat], dtype=np.float32)
        cache[fn] = {
            'conv_shape': img_conv.shape,
            'sat_shape': img_sat.shape,
            'coords_conv': coords_conv,
            'desc_conv': desc_conv,
            'coords_sat': coords_sat,
            'desc_sat': desc_sat
        }
    return cache

def init_worker(cache_path):
    global CACHE
    with open(cache_path, 'rb') as f:
        CACHE = pickle.load(f)
def evaluate_param_set(params):
    ratio_thresh, ransac_reproj, nOctaveLayers, contrastThreshold, edgeThreshold, sigma = params
    sift = cv2.SIFT_create(
        nfeatures=6000,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma
    )
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    distances_between_matches = []
    total_inliers, valid_count = 0, 0
    tested_image_names = []
    for fn, entry in CACHE.items():
        coords_c, desc_c = entry['coords_conv'], entry['desc_conv']
        coords_s, desc_s = entry['coords_sat'], entry['desc_sat']

        knn = flann.knnMatch(desc_c, desc_s, k=2)
        good = [m for m, n in knn if m.distance < ratio_thresh * n.distance]
        if len(good) < 4:
            continue
        idx_c = np.array([m.queryIdx for m in good], dtype=int)
        idx_s = np.array([m.trainIdx for m in good], dtype=int)
        pts_conv = coords_c[idx_c].reshape(-1, 1, 2)
        pts_sat = coords_s[idx_s].reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts_conv, pts_sat, cv2.RANSAC, ransac_reproj)
        if H is None:
            continue
        pts_conv_transformed = cv2.perspectiveTransform(pts_conv, H)
        matched_distances = np.linalg.norm(pts_conv_transformed - pts_sat, axis=2).flatten()
        avg_pixel_offset = np.mean(matched_distances[mask.ravel().astype(bool)])
        distances_between_matches.append(avg_pixel_offset)
        total_inliers += int(mask.ravel().sum())
        valid_count += 1
        tested_image_names.append(fn)
    avg_pixel_offset = np.mean(distances_between_matches) if valid_count else float('inf')
    avg_inliers = total_inliers / valid_count if valid_count else 0
    result = {
        'ratio_thresh': ratio_thresh,
        'ransac_reproj': ransac_reproj,
        'nOctaveLayers': nOctaveLayers,
        'contrastThreshold': contrastThreshold,
        'edgeThreshold': edgeThreshold,
        'sigma': sigma,
        'avg_pixel_offset': avg_pixel_offset,
        'avg_inliers': avg_inliers,
        'valid_images': valid_count
    }

    with lock:
        results_file = 'grid_search_results.csv'
        images_file = 'tested_images.csv'
        write_results_header = not os.path.exists(results_file) or os.path.getsize(results_file) == 0
        write_images_header = not os.path.exists(images_file) or os.path.getsize(images_file) == 0
        with open(results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if write_results_header:
                writer.writeheader()
            writer.writerow(result)
        with open(images_file, 'a', newline='') as f_img:
            writer_img = csv.writer(f_img)
            if write_images_header:
                writer_img.writerow(['ratio_thresh', 'ransac_reproj', 'nOctaveLayers', 
                                     'contrastThreshold', 'edgeThreshold', 'sigma', 'tested_images'])
            writer_img.writerow([
                ratio_thresh, ransac_reproj, nOctaveLayers,
                contrastThreshold, edgeThreshold, sigma,
                ';'.join(tested_image_names)
            ])
    return result

def main():
    converted_folder = 'converted_2'
    satellite_folder = 'split_dataset/val/B'
    cache_file = 'sift_cache.pkl'
    image_subset_file = 'image_subset.pkl'
    if os.path.exists(image_subset_file):
        with open(image_subset_file, 'rb') as f:
            subset = pickle.load(f)
    else:
        all_files = sorted(os.listdir(converted_folder))
        subset_size = max(1, int(len(all_files) * 0.27))
        subset = random.sample(all_files, subset_size)
        with open(image_subset_file, 'wb') as f:
            pickle.dump(subset, f)
    print("Images Loaded")
    base_sift = cv2.SIFT_create(nfeatures=5000)
    cache = build_cache(converted_folder, satellite_folder, subset, base_sift)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print("Cache Loaded")
    params_grid = list(itertools.product(
        [0.8,0.825,0.85,0.875,9,9.25,9.5],
        [20, 30, 40,50],
        [10,20, 30, 40,50],
        [0.02, 0.025, 0.03,0.035, 0.04],
        [1, 5, 10, 20,30],
        [1, 1.5, 2,2.5,3]
    ))
    total_combinations = len(params_grid)
    processed_params = set()
    if os.path.exists('grid_search_results.csv'):
        with open('grid_search_results.csv', 'r') as f:
            reader = csv.DictReader(f)
            processed_params = set(tuple(float(row[param]) for param in reader.fieldnames[:6]) for row in reader)
    before_removal = len(params_grid)
    params_grid = [p for p in params_grid if p not in processed_params]
    removed_combinations = before_removal - len(params_grid)
    print(f'Processing {len(params_grid)} out of {total_combinations} total combinations.')

    with Pool(16, initializer=init_worker, initargs=(cache_file,)) as pool:
        list(tqdm(pool.imap_unordered(evaluate_param_set, params_grid), total=len(params_grid), desc='Grid search'))

if __name__ == '__main__':
    main()
