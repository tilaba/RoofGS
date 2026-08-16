#!/bin/bash


cd submodules/diff-gaussian-rasterization 
rm -rf build 
pip install .
cd ../../
python render.py -m ../gaussian-splatting/output/db/playroom  -s ../gaussian-splatting/data/db/playroom  --compute_cov3D_python --skip_train --eval
python metrics.py -m ../gaussian-splatting/output/db/playroom 

# cd submodules/diff-gaussian-rasterization 
# rm -rf build 
# pip install .
# cd ../../
# python -u render.py -m ../gaussian-splatting/output/db/playroom  -s  ../gaussian-splatting/data/db/playroom  --convert_SHs_python   --compute_cov3D_python


# # # # Define the datasets and their respective scenes
# # # # Format: "dataset_name:scene1 scene2 scene3..."
# DATASETS=(
#     "m360:bicycle bonsai counter flowers garden kitchen room stump treehill"
#     "tandt:train truck"
#     "db:drjohnson playroom"
# )

# # # Base paths
# MODEL_BASE="../gaussian-splatting/output"
# # MODEL_BASE="/home/yluo/gaussian-splatting-pup/experiments/train_80_50"
# DATA_BASE="../gaussian-splatting/data"

# for ENTRY in "${DATASETS[@]}"; do
#     # Split the entry into dataset name and scenes
#     DATASET_NAME="${ENTRY%%:*}"
#     SCENES_STRING="${ENTRY#*:}"
#     read -ra SCENES <<< "$SCENES_STRING"

#     echo "================================================"
#     echo " Processing Dataset: $DATASET_NAME"
#     echo "================================================"

#     for SCENE in "${SCENES[@]}"; do
#         MODEL_PATH="$MODEL_BASE/$DATASET_NAME/$SCENE"
#         DATA_PATH="$DATA_BASE/$DATASET_NAME/$SCENE"

#         echo "------------------------------------------------"
#         echo "Evaluating scene: $SCENE ($DATASET_NAME)"
#         echo "------------------------------------------------"

#         # 1. Run rendering
#         # python -u render.py \
#         #     -m "$MODEL_PATH" \
#         #     -s "$DATA_PATH" \
#         #     --compute_cov3D_python \
#             # --resolution 1920

#         python -u render.py \
#             -m "$MODEL_PATH" \
#             -s "$DATA_PATH" \
#             --compute_cov3D_python \
#             # --resolution 1920

#         # python render.py \
#         #     -m "$MODEL_PATH" \
#         #     -s "$DATA_PATH" \
#         #     --skip_train \
#         #     --eval \
#         #     --compute_cov3D_python

#         # python metrics.py \
#         #     -m "$MODEL_PATH"

#         echo "Done with $SCENE"
#     done
# done


