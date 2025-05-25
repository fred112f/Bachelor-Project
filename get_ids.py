import os

folder = "screenshots_SE_Filtered"
ids = []

for filename in os.listdir(folder):
    if filename.startswith("screenshot_") and filename.endswith(".png"):
        number_part = filename[len("screenshot_"):-4] 
        try:
            feature_id = int(number_part)
            ids.append(feature_id)
        except ValueError:
            pass
print(",".join(map(str, ids)))
