import cv2
import numpy as np
import os
import pickle
import pandas as pd

folder_a = 'converted_2'
folder_b = 'split_dataset/val/B'
selected_images_file = 'selected_images.pkl'
config_file = 'filtered_output_after_tuning.csv'
output_root = 'matches_output'

os.makedirs(output_root, exist_ok=True)
with open(selected_images_file, 'rb') as f:
    selected_images = pickle.load(f)
config_idx = 12  
config = pd.read_csv(config_file).iloc[config_idx]
config_folder = os.path.join(output_root, f'config_{config_idx}')
os.makedirs(config_folder, exist_ok=True)
ratio_thresh = config['ratio_thresh']
ransac_reproj_threshold = config['ransac_reproj']
nfeatures = int(config.get('nfeatures', 0))
nOctaveLayers = int(config['nOctaveLayers'])
contrastThreshold = float(config['contrastThreshold'])
edgeThreshold = float(config['edgeThreshold'])
sigma = float(config['sigma'])
sift = cv2.SIFT_create(
    nfeatures=nfeatures,
    nOctaveLayers=nOctaveLayers,
    contrastThreshold=contrastThreshold,
    edgeThreshold=edgeThreshold,
    sigma=sigma
)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
offsets = []
image_names = []
for i, img_name in enumerate(selected_images):
    img1_gray = cv2.imread(os.path.join(folder_a, img_name), cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.imread(os.path.join(folder_b, img_name), cv2.IMREAD_GRAYSCALE)
    img1_color = cv2.imread(os.path.join(folder_a, img_name), cv2.IMREAD_COLOR)
    img2_color = cv2.imread(os.path.join(folder_b, img_name), cv2.IMREAD_COLOR)
    if img1_gray is None or img2_gray is None or img1_color is None or img2_color is None:
        continue

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None:
        continue

    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    if len(good_matches) < 4:
        print(f"Not enough matches for {img_name}")
        continue
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
    matchesMask = mask.ravel().tolist() if mask is not None else None
    pixel_offset = None
    if M is not None:
        transformed_pts = cv2.perspectiveTransform(src_pts, M)
        pixel_offset = np.mean(np.linalg.norm(transformed_pts - dst_pts, axis=2))
        offsets.append(pixel_offset)
        image_names.append(img_name)
    match_img = cv2.drawMatches(
        img1_color, kp1, img2_color, kp2, good_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    output_path = os.path.join(config_folder, f'match_{i}_{img_name}')
    cv2.imwrite(output_path, match_img)

offset_df = pd.DataFrame({'image': image_names, 'pixel_offset': offsets})
offset_df.loc[len(offset_df.index)] = ['average', np.mean(offsets) if offsets else float('inf')]
offset_df.to_csv(os.path.join(config_folder, 'offsets.csv'), index=False)
