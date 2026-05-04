# modelscope download --dataset OpenDataLab/Hypersim --local_dir '/path/to/datasets/hypersim_depth' \
#  --include '/raw/'

list=(
# ai_030_008
# ai_031_010
# ai_031_004
# ai_041_003
# ai_017_007
# ai_015_004
# ai_035_003
# ai_004_003
# ai_009_007
# ai_004_010
# ai_003_010
# ai_045_008
# ai_028_008
# ai_052_001
# ai_038_009
# ai_010_007
# ai_018_005
# ai_019_008
# ai_009_009
# ai_022_010
# ai_007_001
# ai_015_001
# ai_053_003
# ai_055_009
ai_053_005
ai_052_007
ai_023_003
ai_046_005
ai_005_005
ai_006_007
ai_052_003
ai_050_002
ai_047_009
ai_024_012
ai_016_001
ai_048_001
ai_044_003
ai_039_003
ai_012_005
ai_035_004
ai_044_001
ai_004_004
ai_016_004
ai_051_004
ai_024_013
ai_032_007)
for i in ${list[@]}; do
    echo $i
    modelscope download --dataset OpenDataLab/Hypersim --local_dir '/path/to/datasets/hypersim_depth/' \
     --include "raw/${i}.zip"
done

# ai_030_008
# ai_031_010
# ai_031_004
# ai_041_003
# ai_017_007
# ai_015_004
# ai_035_003
# ai_004_003
# ai_009_007
# ai_004_010
# ai_003_010
# ai_045_008
# ai_028_008
# ai_052_001
# ai_038_009
# ai_010_007
# ai_018_005
# ai_019_008
# ai_009_009
# ai_022_010
# ai_007_001
# ai_015_001
# ai_053_003
# ai_055_009
# ai_053_005
# ai_052_007
# ai_023_003
# ai_046_005
# ai_005_005
# ai_006_007
# ai_052_003
# ai_050_002
# ai_047_009
# ai_024_012
# ai_016_001
# ai_048_001
# ai_044_003
# ai_039_003
# ai_012_005
# ai_035_004
# ai_044_001
# ai_004_004
# ai_016_004
# ai_051_004
# ai_024_013
# ai_032_007