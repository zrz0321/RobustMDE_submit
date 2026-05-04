import os

file_path = "external/Depth_Anything_V2/metric_depth/dataset/splits/hypersim/val.txt"
unique_pack = set()
with open(file_path, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    for path in lines:
        # identify the last forth segment
        segments = path.split('/')
        if len(segments) < 4:
            continue
        pack = segments[-4]
        unique_pack.add(pack)
        
for pack in unique_pack:
    print(pack)
print(f"Total unique packs: {len(unique_pack)}")