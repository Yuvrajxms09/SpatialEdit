# SpatialEdit: Benchmarking Fine-Grained Image Spatial Editing

<p align="center">
  Yicheng Xiao*, Wenhu Zhang*, Lin Song ✉️, Yukang Chen, Wenbo Li, Nan Jiang, Tianhe Ren, Haokun Lin, Wei Huang, Haoyang Huang, Xiu Li, Nan Duan, Xiaojuan Qi ✉️
</p>

<p align="center">
  🧭 Fine-grained spatial editing &nbsp;|&nbsp; 🧪 Benchmarking &nbsp;|&nbsp; 🎥 Camera and object manipulation
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2505.13031"><img src='https://img.shields.io/badge/MindOmni-Paper-red?logo=bookstack&logoColor=red'></a>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/jdopensource/JoyAI-Image-Edit/tree/main/JoyAI-Image-Edit">Training Data</a> &nbsp;•&nbsp;
  🧠 <a href="https://huggingface.co/jdopensource/JoyAI-Image-Edit/tree/main/JoyAI-Image-Edit">Model Weights</a> &nbsp;•&nbsp;
  🖼️ <a href="https://huggingface.co/jdopensource/JoyAI-Image-Edit/tree/main/JoyAI-Image-Edit">Benchmark Images</a>
</p>

## 📝 Abstract

Image spatial editing performs geometry-driven transformations, allowing precise control over object layout and camera viewpoints. Current models are insufficient for fine-grained spatial manipulations, motivating a dedicated assessment suite.

Our contributions are three-fold:

1. We introduce **SpatialEdit-Bench**, a complete benchmark that evaluates spatial editing by jointly measuring perceptual plausibility and geometric fidelity via viewpoint reconstruction and framing analysis.
2. To address the data bottleneck for scalable training, we construct **SpatialEdit-500K**, a synthetic dataset generated with a controllable Blender pipeline that renders objects across diverse backgrounds and systematic camera trajectories, providing precise ground-truth transformations for both object- and camera-centric operations.
3. Building on this data, we develop **SpatialEdit-16B**, a baseline model for fine-grained spatial editing. Our method achieves competitive performance on general editing while substantially outperforming prior methods on spatial manipulation tasks.

## 🔗 Resources

| Resource | Description | Link |
| --- | --- | --- |
| 🧪 Training Data | SpatialEdit-500K synthetic training set for scalable fine-grained spatial editing | 🤗[Hugging Face](https://huggingface.co/jdopensource/JoyAI-Image-Edit/tree/main/JoyAI-Image-Edit) |
| 🧠 Model Weights | SpatialEdit-16B checkpoints for image spatial editing | 🤗[Hugging Face](https://huggingface.co/jdopensource/JoyAI-Image-Edit/tree/main/JoyAI-Image-Edit) |
| 🖼️ Benchmark Images | SpatialEdit-Bench benchmark images and evaluation assets | 🤗[Hugging Face](https://huggingface.co/jdopensource/JoyAI-Image-Edit/tree/main/JoyAI-Image-Edit) |

## 🌍 Overview

SpatialEdit focuses on spatially grounded image editing, where the goal is not just to change appearance, but to control object motion, rotation, 3D viewpoint, framing, and camera movement with precision.

![Task Definition](assets/task_definition.png)

## 📏 SpatialEdit-Bench

SpatialEdit-Bench evaluates both object-centric and camera-centric edits. The benchmark is designed to score whether an edited image is visually plausible while also satisfying the requested spatial transformation.

![SpatialEdit-Bench Results](assets/spatialedit_bench_result.png)

## 🏗️ SpatialEdit-500K Data Engine

To support scalable training and controlled evaluation, SpatialEdit-500K is built with a synthetic rendering pipeline that systematically varies object pose, placement, and camera trajectories over diverse scenes.

![SpatialEdit-500K Data Engine](assets/data_engine.png)

## 🎨 Visual Comparisons

Qualitative comparisons highlight the advantage of SpatialEdit on fine-grained spatial manipulation tasks.

![Visual Comparison 1](assets/visual_compare1.png)

![Visual Comparison 2](assets/visual_compare2.png)

## 🚀 Application Gallery

### 🧊 3D Point Control

<p align="center">
  <img src="assets/application/3dpoint/01.gif" width="23%" />
  <img src="assets/application/3dpoint/02.gif" width="23%" />
  <img src="assets/application/3dpoint/11.gif" width="23%" />
  <img src="assets/application/3dpoint/12.gif" width="23%" />
</p>

✨ The first and third examples show point clouds with only a single given viewpoint. The second and fourth examples are augmented by our model, which synthesizes richer spatial observations from the sparse input view.

### 🎥 Camera Trajectory Editing

<p align="center">
  <img src="assets/application/camera/input.png" width="31%" />
  <img src="assets/application/camera/output.png" width="31%" />
  <img src="assets/application/camera/video.gif" width="31%" />
</p>

✨ Given the first frame, our editing model first produces the target second frame, and a video generation model then synthesizes an engaging camera-transition video between them while preserving scene realism and subject consistency.

### 🚶 Object Translation

<p align="center">
  <img src="assets/application/moving/input.png" width="31%" />
  <img src="assets/application/moving/output.png" width="31%" />
  <img src="assets/application/moving/video.gif" width="31%" />
</p>

✨ Given the first frame, our editing model first generates the target second frame, and a video generation model then produces a coherent motion sequence between the two frames while keeping the scene layout and camera setup stable.

### 🔄 Object Rotation

<p align="center">
  <img src="assets/application/rotation/input.png" width="31%" />
  <img src="assets/application/rotation/output.png" width="31%" />
  <img src="assets/application/rotation/video.gif" width="31%" />
</p>

✨ Given the first frame, our editing model first generates the target second frame, and a video generation model then creates a smooth rotational transition between them while maintaining environmental consistency.

## ⚙️ Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
pip install accelerate peft gradio pillow
```

Notes:

- `flash_attn` in `requirements.txt` requires a compatible CUDA and PyTorch environment.
- Some config files still contain placeholder or internal paths and should be updated before running inference.

## 🧪 Quick Demo

The repo currently provides a simple local inference entry point:

```bash
python spatialedit_demo.py
```

Before running, update the checkpoint paths in `spatialedit_demo.py`:

- `ckpt_path_PT`
- `ckpt_path_CT`
- `device`

The example input image is located at `validation/JD_Dog.jpeg`.

## 🏃 Benchmark Inference

To generate edited outputs for SpatialEdit-Bench, use:

```bash
torchrun --nnodes 1 --nproc_per_node 8 SpatialEdit-Bench/eval_inference.py \
  --config configs/spatialedit_base_config.py \
  --ckpt-path /path/to/checkpoint_or_lora \
  --save-path /path/to/save_dir \
  --meta-file /path/to/SpatialEdit_Bench_Meta_File.json \
  --bench-data-dir /path/to/SpatialEdit_Bench_Data \
  --basesize 1024 \
  --num-inference-steps 50 \
  --guidance-scale 5.0 \
  --seed 42
```

You can also adapt the provided launcher script:

- `SpatialEdit-Bench/scripts/dist_inference.sh`

## 📊 Benchmark Evaluation

### 📷 Camera-Level Evaluation

Camera-level evaluation measures viewpoint reconstruction and framing fidelity:

```bash
bash SpatialEdit-Bench/scripts/dist_camera_eval.sh
```

Update the placeholder paths in the script before running:

- `VGGT`
- `YOLO`
- `EVAL_DATA`
- `META_DATA_FILE`

### 🧩 Object-Level Evaluation

Object-level evaluation scores edit faithfulness and benchmark statistics:

```bash
bash SpatialEdit-Bench/scripts/dist_object_eval.sh
```

Update the script paths and evaluation backend first:

- `META_FILE`
- `SAVE`
- `BENCH_DATA_DIR`
- `BACKBONE`

## 💡 Notes

- `configs/spatialedit_base_config.py` currently contains internal absolute paths and should be replaced with your local model paths.
- The benchmark scripts assume access to external benchmark metadata, source images, and model checkpoints.
- The repo already includes example evaluation utilities under `SpatialEdit-Bench/camera_level_eval` and `SpatialEdit-Bench/object_level_eval`.

## ❤️ Acknowledgement

Code in this repository is built upon several public repositories. Thanks for the wonderful work [recamaster](https://github.com/576218404/RecamMaster) and [TexVerse](https://github.com/ffiirree/TexVerse) ! !

## 🎬 Demo

The following demo showcases our method on fine-grained spatial editing and video generation from spatially controlled endpoints.

<p align="center">
  <img src="assets/demo.gif" width="88%" />
</p>

Our data construction pipeline and benchmark experiments are built on top of the image editing model from [JoyAI-Image](https://github.com/jd-opensource/JoyAI-Image).
