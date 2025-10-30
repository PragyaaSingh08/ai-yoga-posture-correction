import pandas as pd
df = pd.read_csv("outputs/angles_pose_keypoints.csv")
print(df['pose'].unique())
