import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle

# ==================== config ====================
ROOT_DIR = r'C:\迅雷云盘\main\NuclearPowerPlantAccidentData-main\Operation_csv_data'
TIME_STEPS = 150
EXPECTED_COLS = 96
SEED = 42

# ==================== load data ====================
print("Loading CSV files...")

samples = []
labels = []
skipped = []

accident_types = sorted([d for d in os.listdir(ROOT_DIR) 
                         if os.path.isdir(os.path.join(ROOT_DIR, d))])

for label_idx, accident in enumerate(accident_types):
    accident_path = os.path.join(ROOT_DIR, accident)
    csv_files = [f for f in os.listdir(accident_path) 
                 if f.endswith('.csv') and 'dose' not in f.lower()]
    loaded = 0
    
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(os.path.join(accident_path, csv_file))
            if len(df.columns) != EXPECTED_COLS + 1:
                skipped.append(f"{accident}/{csv_file}: {len(df.columns)} cols")
                continue
            
            data = df.iloc[:TIME_STEPS, 1:].values.astype(np.float32)
            if data.shape[0] == 0:
                skipped.append(f"{accident}/{csv_file}: empty")
                continue
            
            if data.shape[0] < TIME_STEPS:
                pad = np.zeros((TIME_STEPS - data.shape[0], EXPECTED_COLS), dtype=np.float32)
                data = np.vstack([data, pad])
            
            df_temp = pd.DataFrame(data)
            df_temp = df_temp.interpolate(method='linear', limit_direction='both').fillna(0)
            data = df_temp.values.astype(np.float32)
            
            if data.shape != (TIME_STEPS, EXPECTED_COLS):
                skipped.append(f"{accident}/{csv_file}: shape {data.shape}")
                continue
            
            samples.append(data)
            labels.append(label_idx)
            loaded += 1
            
        except Exception as e:
            skipped.append(f"{accident}/{csv_file}: {e}")
    
    print(f"  [{accident}] {loaded}/{len(csv_files)} loaded")

samples = np.array(samples)
labels = np.array(labels)

print(f"\nTotal loaded: {len(samples)} samples")
print(f"Skipped: {len(skipped)}")

# class histogram
class_counts = Counter(labels)
print("\nClass distribution:")
for i, name in enumerate(accident_types):
    print(f"  {name}: {class_counts[i]} samples")

# ==================== standardize ====================
print("\nNormalizing...")

N, T, F = samples.shape
samples_2d = samples.reshape(-1, F)
scaler = StandardScaler()
samples_2d = scaler.fit_transform(samples_2d)
samples = samples_2d.reshape(N, T, F)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ==================== split (handle rare classes) ====================
print("\nSplitting...")

# rare classes: fewer than 3 samples
rare_classes = [cls for cls, cnt in class_counts.items() if cnt < 3]
print(f"Rare classes (< 3 samples): {[accident_types[c] for c in rare_classes]}")

# separate rare vs abundant classes
rare_indices = [i for i, y in enumerate(labels) if y in rare_classes]
abundant_indices = [i for i, y in enumerate(labels) if y not in rare_classes]

X_rare = samples[rare_indices]
y_rare = labels[rare_indices]
X_abundant = samples[abundant_indices]
y_abundant = labels[abundant_indices]

print(f"Abundant: {len(X_abundant)} | Rare: {len(X_rare)}")

# abundant classes: stratified 70 / 15 / 15
X_temp, X_test, y_temp, y_test = train_test_split(
    X_abundant, y_abundant, test_size=0.15, random_state=SEED, stratify=y_abundant
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=SEED, stratify=y_temp
)

# put every rare-class sample in train so the model sees it
if len(X_rare) > 0:
    X_train = np.vstack([X_train, X_rare])
    y_train = np.concatenate([y_train, y_rare])

print(f"Final -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ==================== FFT ====================
print("Computing FFT...")

def compute_fft(x):
    fft_vals = np.fft.rfft(x, axis=0)
    fft_mag = np.abs(fft_vals)
    fft_full = np.zeros((T, F), dtype=np.float32)
    fft_full[:len(fft_mag)] = fft_mag
    return fft_full

X_train_fft = np.array([compute_fft(x) for x in X_train])
X_val_fft   = np.array([compute_fft(x) for x in X_val])
X_test_fft  = np.array([compute_fft(x) for x in X_test])

# ==================== save ====================
print("Saving...")

np.save('train.npy', X_train)
np.save('val.npy', X_val)
np.save('test.npy', X_test)
np.save('train_labels.npy', y_train)
np.save('val_labels.npy', y_val)
np.save('test_labels.npy', y_test)

np.save('train_fft.npy', X_train_fft)
np.save('val_fft.npy', X_val_fft)
np.save('test_fft.npy', X_test_fft)

with open('class_names.txt', 'w') as f:
    for i, name in enumerate(accident_types):
        f.write(f"{i}: {name}\n")

print("\nDone!")