import pandas as pd
import os

for split in ['train_split.csv', 'val_split.csv', 'test_split.csv']:
    df = pd.read_csv(split, header=None, names=['filename','label'])
    missing = [f for f in df['filename'] if not os.path.exists(os.path.join('data', f))]
    print(f'{split}: {len(missing)} missing files')
    if missing:
        print(f'  Examples: {missing[:5]}')