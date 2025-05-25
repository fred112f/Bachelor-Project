import os
import glob
import cv2
import numpy as np
import multiprocessing
import random

def largest_connected_component(bin_mask):
    bin_mask_8u = bin_mask.astype("uint8")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask_8u, connectivity=8)
    if num_labels <= 1:
        return bin_mask_8u  
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype("uint8")
    return largest_mask

def random_polygon_path_8dir(bin_mask, 
                             num_segments=6, 
                             max_segment_length=50):
    coords = np.argwhere(bin_mask == 1)
    if len(coords) == 0:
        return []
    start_index = random.randrange(len(coords))
    start_r, start_c = coords[start_index]
    path = [(start_r, start_c)]
    visited_mask = np.zeros_like(bin_mask, dtype=np.uint8)
    visited_mask[start_r, start_c] = 1
    directions8 = [
        (-1, -1), 
        (-1,  0),  
        (-1,  1), 
        ( 0,  1),  
        ( 1,  1),  
        ( 1,  0),  
        ( 1, -1),  
        ( 0, -1),  
    ]
    current_dir_idx = random.randrange(len(directions8))
    def try_step_visited(r, c, dr, dc):
        nr, nc = r + dr, c + dc
        if 0 <= nr < bin_mask.shape[0] and 0 <= nc < bin_mask.shape[1]:
            if bin_mask[nr, nc] == 1 and visited_mask[nr, nc] == 0:
                return nr, nc
        return None
    def try_step_ignore_visited(r, c, dr, dc):
        nr, nc = r + dr, c + dc
        if 0 <= nr < bin_mask.shape[0] and 0 <= nc < bin_mask.shape[1]:
            if bin_mask[nr, nc] == 1:
                return nr, nc
        return None
    for _ in range(num_segments - 1):
        length = random.randint(5, max_segment_length)
        row, col = path[-1]
        dr, dc = directions8[current_dir_idx]
        new_r, new_c = row, col
        for _step in range(length):
            nxt = try_step_visited(new_r, new_c, dr, dc)
            if not nxt:
                break
            new_r, new_c = nxt
            visited_mask[new_r, new_c] = 1
        if (new_r, new_c) == (row, col):
            break
        path.append((new_r, new_c))
        possible_shifts = [-3, -2, -1, 1, 2, 3]
        weights = [0.05, 0.4, 0.05, 0.05, 0.4, 0.05]
        chosen_shift = random.choices(possible_shifts, weights=weights, k=1)[0]
        current_dir_idx = (current_dir_idx + chosen_shift) % 8
    if len(path) < 2:
        return []

    last_r, last_c = path[-1]
    first_r, first_c = path[0]
    close_path = []
    if random.random() < 0.5:
        col_dir = 1 if (first_c > last_c) else -1
        while last_c != first_c:
            nxt = try_step_ignore_visited(last_r, last_c, 0, col_dir)
            if not nxt:
                return []
            last_r, last_c = nxt
            close_path.append((last_r, last_c))
        row_dir = 1 if (first_r > last_r) else -1
        while last_r != first_r:
            nxt = try_step_ignore_visited(last_r, last_c, row_dir, 0)
            if not nxt:
                return []
            last_r, last_c = nxt
            close_path.append((last_r, last_c))
    else:
        row_dir = 1 if (first_r > last_r) else -1
        while last_r != first_r:
            nxt = try_step_ignore_visited(last_r, last_c, row_dir, 0)
            if not nxt:
                return []
            last_r, last_c = nxt
            close_path.append((last_r, last_c))
        col_dir = 1 if (first_c > last_c) else -1
        while last_c != first_c:
            nxt = try_step_ignore_visited(last_r, last_c, 0, col_dir)
            if not nxt:
                return []
            last_r, last_c = nxt
            close_path.append((last_r, last_c))
    if (last_r, last_c) != (first_r, first_c):
        return []
    path.extend(close_path)
    return path

def lighten_color(bgr, increment=80):
    return tuple(min(channel + increment, 255) for channel in bgr)
def draw_polygon(img, path, outline_color, do_fill=False, fill_color=None):
    if len(path) < 3:
        return
    outline_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i in range(1, len(path)):
        r1, c1 = path[i - 1]
        r2, c2 = path[i]
        pt1 = (c1, r1)  
        pt2 = (c2, r2)
        cv2.line(outline_mask, pt1, pt2, 255, thickness=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    dilated_outline = cv2.dilate(outline_mask, kernel, iterations=1)
    outline_pixels = np.where(dilated_outline > 0)
    img[outline_pixels[0], outline_pixels[1], :] = outline_color
    if do_fill and fill_color is not None:
        fill_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        polygon_points = []
        for (r, c) in path:
            polygon_points.append([c, r])
        polygon_points = np.array([polygon_points], dtype=np.int32)
        cv2.fillPoly(fill_mask, polygon_points, 255)
        fill_pixels = np.where(fill_mask == 255)
        img[fill_pixels[0], fill_pixels[1], :] = fill_color

def process_single_image(args):
    file_name, source_folder, output_folder = args
    img_path = os.path.join(source_folder, file_name)
    img = cv2.imread(img_path)
    if img is None:
        return
    center_bgr = np.array([223, 211, 170], dtype=np.uint8)
    tol = 10
    lower_bgr = np.clip(center_bgr - tol, 0, 255).astype(np.uint8)
    upper_bgr = np.clip(center_bgr + tol, 0, 255).astype(np.uint8)
    mask = cv2.inRange(img, lower_bgr, upper_bgr)
    largest_mask = largest_connected_component(mask)
    if np.count_nonzero(largest_mask) == 0:
        return
    n_polygons = random.randint(1, 4)
    color_choices = [
        (0, 255, 0),    
        (255, 0, 0),   
        (0, 0, 255),   
        (0, 255, 255),  
    ]
    for _ in range(n_polygons):
        segs = random.randint(4, 16) 
        max_len = random.randint(80, 300)  
        path = random_polygon_path_8dir(
            largest_mask, 
            num_segments=segs, 
            max_segment_length=max_len
        )
        if len(path) < 4:
            continue
        outline_color = random.choice(color_choices)
        do_fill = (random.random() < 0.5)
        fill_col = None
        if do_fill:
            fill_col = lighten_color(outline_color, increment=60)
        draw_polygon(img, path, outline_color, do_fill=do_fill, fill_color=fill_col)
    out_path = os.path.join(output_folder, file_name)
    cv2.imwrite(out_path, img)
    print(f"Processed and saved: {out_path}")

def main():
    source_folder = "replace_color/final_training"
    output_folder = "replace_color/train/A"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_files = glob.glob(os.path.join(source_folder, "*.*"))
    filenames = [os.path.basename(f) for f in image_files]
    args_list = [(fname, source_folder, output_folder) for fname in filenames]
    with multiprocessing.Pool(processes=24) as pool:
        pool.map(process_single_image, args_list)

if __name__ == "__main__":
    main()
