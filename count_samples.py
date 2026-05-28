import os

data_dir = "data"
total = 0
for d in sorted(os.listdir(data_dir)):
    full = os.path.join(data_dir, d)
    if os.path.isdir(full):
        npy_files = [f for f in os.listdir(full) if f.endswith(".npy")]
        count = len(npy_files)
        total += count
        print(f"{d}: {count} files")
        if count > 0:
            import numpy as np
            sample = np.load(os.path.join(full, npy_files[0]))
            print(f"  Sample shape: {sample.shape}, dtype: {sample.dtype}")
print(f"\nTotal samples: {total}")
