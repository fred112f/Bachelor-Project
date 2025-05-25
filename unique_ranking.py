import pandas as pd

def rank_heuristic(df, weight_avg_pixel_offset, weight_valid_images):
    df['rank'] = df['avg_pixel_offset'] * weight_avg_pixel_offset + df['valid_images'] * weight_valid_images
    sorted_df = df.sort_values(by='rank', ascending=True).reset_index(drop=True)
    return sorted_df

def filter_similar_ranks(df, threshold):
    df_sorted = df.sort_values(by='rank').reset_index(drop=True)
    filtered_rows = [df_sorted.iloc[0]]
    for _, row in df_sorted.iterrows():
        if abs(row['rank'] - filtered_rows[-1]['rank']) > threshold:
            filtered_rows.append(row)
    return pd.DataFrame(filtered_rows).reset_index(drop=True)

def main():
    df = pd.read_csv('deduplicated_file.csv')
    weight_avg_pixel_offset = 0.6
    weight_valid_images = 0.4
    rank_threshold = 10
    ranked_df = rank_heuristic(df, weight_avg_pixel_offset, weight_valid_images)
    filtered_df = filter_similar_ranks(ranked_df, rank_threshold)
    filtered_df.to_csv('filtered_output.csv', index=False)
if __name__ == "__main__":
    main()
