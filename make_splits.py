import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Build list of all images and labels
data_path = 'data'
files, labels = [], []

for fname in os.listdir(data_path):
    if fname.endswith('.png'):
        files.append(fname)
        labels.append(0 if fname.startswith('Normal') else 1)

df = pd.DataFrame({'filename': files, 'label': labels})
print(f"Total images: {len(df)}")
print(f"Normal: {(df.label==0).sum()}  TB: {(df.label==1).sum()}")

# 80% train, 10% val, 10% test — stratified so both classes are balanced
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

train_df.to_csv('train_split_new.csv', index=False, header=False)
val_df.to_csv('val_split_new.csv',     index=False, header=False)
test_df.to_csv('test_split_new.csv',   index=False, header=False)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("Splits saved!")