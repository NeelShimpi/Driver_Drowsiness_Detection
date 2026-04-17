import os
from collections import Counter

dataset_path = r"c:\Users\neels\MajorProject\project\driver_drowsiness_dataset\dataset"

print(f"\nScanning: {dataset_path}\n" + "="*50)

valid_exts = {'.jpg', '.jpeg', '.png'}
class_counts = {}
extensions = Counter()
invalid_files = []

for root, dirs, files in os.walk(dataset_path):
    category = os.path.basename(root)
    if root == dataset_path:
        continue
        
    class_counts[category] = 0
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        extensions[ext] += 1
        
        if ext in valid_exts:
            class_counts[category] += 1
        else:
            invalid_files.append(os.path.join(category, f))

print("CLASS COUNTS:")
total = 0
for cat, count in class_counts.items():
    print(f" - {cat}: {count} valid images")
    total += count
print(f" - TOTAL: {total}\n")

print("FILE EXTENSIONS FOUND:")
for ext, count in extensions.items():
    print(f" - {ext}: {count}")

print(f"\nINVALID/UNSUPPORTED FILES ({len(invalid_files)}):")
for f in invalid_files[:5]:
    print(f" - {f}")
if len(invalid_files) > 5:
    print(f"   ... and {len(invalid_files) - 5} more")

print("="*50 + "\n")
