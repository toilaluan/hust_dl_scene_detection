import pandas as pd

full_annotation_file = "dataset/label.xlsx"
image_root = "dataset"

ann = pd.read_excel(full_annotation_file).dropna()

# split train test
from sklearn.model_selection import train_test_split

train_ann, test_ann = train_test_split(ann, test_size=0.2, random_state=42)
train_ann.to_excel("dataset/train_label.xlsx", index=False)
test_ann.to_excel("dataset/test_label.xlsx", index=False)
