import cv2
import numpy as np
import os
from tqdm import tqdm

def sift_ransac_homography(
    img_path1,
    img_path2,
    ratio_thresh=0.75,
    ransac_method=cv2.RANSAC,
    ransac_reproj_threshold=5.0,
    nfeatures=0,
    nOctaveLayers=3,
    contrastThreshold=0.04,
    edgeThreshold=10,
    sigma=1.6
):
    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return None
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma
    )
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_knn = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches_knn if m.distance < ratio_thresh * n.distance]
    if len(good_matches) < 4:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, ransac_method, ransac_reproj_threshold)
    if M is None:
        return None
    inlier_src_pts = src_pts[mask.ravel() == 1]
    inlier_dst_pts = dst_pts[mask.ravel() == 1]
    offsets = np.linalg.norm(inlier_src_pts - inlier_dst_pts, axis=2)
    avg_offset = np.mean(offsets)
    return os.path.basename(img_path1), avg_offset
def process_all_images(folder1, folder2, threshold=10, output_file="below_threshold.txt", **sift_params):
    images = sorted(os.listdir(folder1))
    paths = [(os.path.join(folder1, img), os.path.join(folder2, img)) for img in images]
    results = []
    for path_pair in tqdm(paths, desc="Processing images"):
        result = sift_ransac_homography(*path_pair, **sift_params)
        if result is not None:
            img_name, avg_offset = result
            if avg_offset < threshold:
                results.append(f"{img_name}: Avg offset = {avg_offset:.2f}px")
    with open(output_file, "w") as f:
        for line in results:
            f.write(line + "\n")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    folder1 = "converted_2"
    folder2 = "split_dataset/val/B"
    sift_params = {
        'ratio_thresh': 0.85,
        'ransac_method': cv2.RANSAC,
        'ransac_reproj_threshold': 30,
        'nfeatures': 0,
        'nOctaveLayers': 40,
        'contrastThreshold': 0.04,
        'edgeThreshold': 5,
        'sigma': 1
    }
    process_all_images(folder1, folder2, threshold=10, output_file="below_threshold.txt", **sift_params)