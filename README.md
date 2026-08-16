# RoofGS: Roofline-Guided End-to-End Acceleration of 3D Gaussian Splatting



**Abstract:** *3D Gaussian Splatting (3DGS) enables real-time novel-view synthesis but remains limited on GPUs at high resolutions. Through a stage-wise Roofline characterization, we identify two distinct hardware bottlenecks: global memory traffic dominates the front end, whereas instruction throughput limits rasterization. Guided by this analysis, we develop RoofGS, a rendering framework that applies bottleneck-specific optimizations rather than generic kernel acceleration. For the memory-bound front end, we design a resolution-adaptive quantized depth sorting key that compresses each key to 32 bits. For the compute-bound rasterizer, we introduce a range-aware bit-level fast exponential approximation tailored to the bounded exponent range after opacity culling, with a derived per-pixel error bound. These two core techniques are complemented by other developed optimization methods (kernel fusion, compact attribute storage, culling, dual-pixel evaluation) that further reduce memory traffic and improve ILP. Experiments show that RoofGS achieves a 10.1$\times$ end-to-end speedup over 3DGS at 4K on an RTX 4090, increasing throughput from 61 to 616 FPS, with only a 0.028 dB PSNR loss.*

## Setup
Follow the setup instructions for the original [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) codebase. Our code changes are made in (1) the differential renderer submodule and (2) the Python files in this repo.

## Running

To train and evaluate a scene with Speedy-Splat, set the environment variables `SCENE_DATA_PATH` and `SCENE_MODEL_PATH`, then run:

```shell
bash train.sh
```

* `SCENE_DATA_PATH` is the path to the COLMAP or NeRF Synthetic dataset.

* `SCENE_MODEL_PATH` is the path where the model will be saved.

This script follows the 3D-GS pipeline to train a scene on the data provided by `SCENE_DATA_PATH` and saves the model in the `SCENE_MODEL_PATH` directory. Tensorboard logging has been updated to include all metrics reported in the Speedy-Splat manuscript.

Note: This setup trains at a higher image resolution than the original 3D-GS `full_eval.py` protocol, resulting in larger models and a more demanding evaluation setting.

### Scene Metrics

To compute scene metrics for an already trained model, set the environment variables `SCENE_DATA_PATH`,`SCENE_MODEL_PATH`, and `ONLY_RAW_KERNEL_TIMES` then run:

```shell
bash compute_scene_metrics.sh
```

* `SCENE_DATA_PATH` is the path to the COLMAP or NeRF Synthetic dataset.

* `SCENE_MODEL_PATH` is the path to the pretrained 3D-GS model.

* `ONLY_RAW_KERNEL_TIMES` is a boolean (`true`|`false`) variable. When true, the kernel only returns raw kernel time per image.

This script returns all metrics reported in Speedy-Splat for a pretrained model located in `SCENE_MODEL_PATH` with corresponding data in `SCENE_DATA_PATH`. The metrics will be written to a CSV file located in `SCENE_MODEL_PATH/<train|test>/ours_<iteration>/metrics.csv`, where `train`|`test` is the corresponding dataset split and `iteration` is the model checkpoint iteration.

If `ONLY_RAW_KERNEL_TIMES` is set to `true`. then the script will generate a `kernel_times.csv` file instead in the same directory as `metrics.csv`. Each row in the CSV records the raw kernel time to render the corresponding image in milliseconds. Currently, this script sequentially renders and records each image in the dataset, then repeats this process 20 times.
