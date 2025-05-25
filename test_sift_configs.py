import cv2
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle

folder_a = 'converted_images_fine_tuned'
folder_b = 'split_dataset/val/B'
configs = pd.read_csv('filtered_output_after_tuning.csv')
cache_file = 'evaluation_cache.pkl'
selected_images_file = 'selected_images.pkl'
if os.path.exists(selected_images_file):
    with open(selected_images_file, 'rb') as f:
        selected_images = pickle.load(f)
else:
    random.seed(42)
    image_files = sorted(os.listdir(folder_a))
    selected_images = random.sample(image_files, 10)
    with open(selected_images_file, 'wb') as f:
        pickle.dump(selected_images, f)
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
else:
    cache = {}

def evaluate_config(config):
    config_tuple = tuple(config.items())
    if config_tuple in cache:
        return cache[config_tuple]
    sift = cv2.SIFT_create(
        nfeatures=int(config.get('nfeatures', 0)),
        nOctaveLayers=int(config['nOctaveLayers']),
        contrastThreshold=float(config['contrastThreshold']),
        edgeThreshold=float(config['edgeThreshold']),
        sigma=float(config['sigma'])
    )
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    pixel_offsets = []
    num_matches = []
    num_inliers = []
    for img_name in selected_images:
        img1 = cv2.imread(os.path.join(folder_a, img_name), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(folder_b, img_name), cv2.IMREAD_GRAYSCALE)
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None or des1.shape[0] < 2 or des2.shape[0] < 2:
            continue
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except cv2.error as e:
            continue
        good_matches = []
        for m, n in matches:
            if m.distance < config['ratio_thresh'] * n.distance:
                good_matches.append(m)
        if len(good_matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config['ransac_reproj'])
            if M is not None and mask is not None:
                inliers = int(np.sum(mask)) 
                transformed_pts = cv2.perspectiveTransform(src_pts, M)
                offset = np.mean(np.linalg.norm(transformed_pts - dst_pts, axis=2))
                pixel_offsets.append(offset)
                num_matches.append(len(good_matches))
                num_inliers.append(inliers)
    avg_offset = np.mean(pixel_offsets) if pixel_offsets else float('inf')
    avg_matches = np.mean(num_matches) if num_matches else 0
    avg_inliers = np.mean(num_inliers) if num_inliers else 0
    cache[config_tuple] = (avg_offset, avg_matches, avg_inliers)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    return avg_offset, avg_matches, avg_inliers

for idx, config in configs.iterrows():
    avg_offset, avg_matches, avg_inliers = evaluate_config(config)
    for param, value in config.items():
        print(f"  {param}: {value}")
    print(f"Avg Pixel Offset: {avg_offset:.2f}")
    print(f"Avg Matches: {avg_matches:.2f}")
    print(f"Avg Inliers: {avg_inliers:.2f}")
