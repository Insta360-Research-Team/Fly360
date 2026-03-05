
# Fly360 🚁: Omnidirectional Obstacle Avoidance within Drone View
<p align="center">
  <a href=''><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://zxkai.github.io/fly360/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=insta360&logoColor=white' alt='Project Page'></a>
</p>

Official code release for **“Fly360: Omnidirectional Obstacle Avoidance within Drone View”**.

Fly360 maps 🌐 **360° panoramic RGB (ERP)** to **body-frame velocity commands** via a depth-as-intermediate pipeline and a lightweight spherical-conv recurrent policy.

## ✨ Highlights

- 🧠 **Two-stage pipeline**: panoramic RGB → depth → control policy
- 🧭 **Omnidirectional awareness** for obstacle avoidance beyond forward-view sensing
- ⚡ **Lightweight policy** (SphereConv + GRU) for real-time control

## ⚙️ Installation

```bash
conda create -n fly360 python=3.8 -y
conda activate fly360
pip install -r requirements.txt
```

### AirSim Python client (only needed for AirSim)

If you run the code with AirSim, install the AirSim Python API:

```bash
pip install airsim
```

If you run the code with **AirSim360**, you can skip this step.

## 🧩 Environments

### Option A: AirSim environments

We use the AirSim+UE environments from **AirVLN**. Please follow the official instructions in the AirVLN repository to download the simulators/environments, then place them under an `ENVs/` folder **next to this repo** (i.e., in the same workspace directory).

- AirVLN repo: [AirVLN/AirVLN](https://github.com/AirVLN/AirVLN)

### Option B: AirSim360 platform (recommended)

We also provide an alternative setup and recommend using **AirSim360** to test this code or explore other panoramic + UAV research workflows.

- AirSim360 repo: [Insta360-Research-Team/AirSim360](https://github.com/Insta360-Research-Team/AirSim360)

After installing AirSim360, please follow its homepage instructions to configure the environment, then run:

```bash
python demo_airsim360.py --resume weights/fly360.pth
```

## 🗺️ AirSim environment layout

The scene manager (`AirsimTools/airsim_plugin/AirsimTool.py`) searches for environments under:

```text
<WORKSPACE>/ENVs/**/env_<scene_id>/LinuxNoEditor/AirVLN.sh
```

Example for `--env_scene 1`:

```bash
chmod +x "../ENVs/env_1/LinuxNoEditor/AirVLN.sh"
```

## 📁 Expected directory structure

```text
<WORKSPACE>/
  Fly360/
    demo_airsim.py
    demo_airsim360.py
    weights/
      fly360.pth
    AirsimTools/
      airsim_plugin/
        AirsimTool.py
          ...
    unik3d/
      ...
  ENVs/
    env_1/
      LinuxNoEditor/
        AirVLN.sh
    ...
```

## 🚀 Quickstart (AirSim)

Start the scene manager:

```bash
python AirsimTools/airsim_plugin/AirsimTool.py --gpus 0 --port 30000
```

Run the demo:

```bash
python demo_airsim.py --env Park --env_scene 1 --resume weights/fly360.pth
```

You can also try:

```bash
python demo_airsim360.py --resume weights/fly360.pth
```

## 🧾 Citation

If you find this code useful, please cite:

```bibtex
@article{fly360_2026,
title={Fly360: Omnidirectional Obstacle Avoidance Within Drone View},
author={Xiangkai Zhang and Dizhe Zhang and Lu Qi and Xu Yang and Zhiyong Liu},
archivePrefix={arxiv},
year={2026}
}
```

## 🙏 Acknowledgements

- [AirSim](https://github.com/microsoft/AirSim)
- [UniK3D](https://github.com/lpiccinelli-eth/UniK3D) (code under ./unik3d)
- [AirVLN](https://github.com/AirVLN/AirVLN)
- [AirSim360](https://github.com/Insta360-Research-Team/AirSim360) 
