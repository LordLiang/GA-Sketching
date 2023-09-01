# GA-Sketching: Shape Modeling from Multi-View Sketching with Geometry-Aligned Deep Implicit Functions
![image](https://github.com/LordLiang/GA-Sketching/blob/main/fig_teaser.png)
## Introduction
Sketch-based shape modeling aims to bridge the gap between 2D drawing and 3D modeling by providing an intuitive and accessible approach to create 3D shapes from 2D sketches. However, existing methods still suffer from limitations in reconstruction quality and multi-view interaction friendliness, hindering their practical application. This paper proposes a faithful and user-friendly iterative solution to tackle these limitations by learning geometry-aligned deep implicit functions from one or multiple sketches. Our method lifts 2D sketches to volume-based feature tensors, which align strongly with the output 3D shape, enabling accurate reconstruction and faithful editing. Such a geometry-aligned feature encoding technique is well-suited to iterative modeling since features from different viewpoints can be easily memorized or aggregated. Based on these advantages, we design a unified interactive system for sketch-based shape modeling. It enables users to generate the desired geometry iteratively by drawing sketches from any number of viewpoints. In addition, it allows users to edit the generated surface by making few local modifications.
## Setup
The code is tested on Ubuntu 18.04 with PyTorch 1.12.1 CUDA 11.6 installed. Please follow the following steps to install PyTorch and PyTorch3D first. All experiments are run on a single NVIDIA GeForce RTX 2080 Ti gpu.
```
# create and activate the conda environment
conda create -n gas python=3.9.15
conda activate gas

# install necessary packages
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d

pip install -r requirements.txt

cd tools/libmesh/
python setup.py build_ext --inplace
```
## Data generation
First you can obtain watertight and simplified meshes via https://github.com/davidstutz/mesh-fusion and put them into 'datasets/shapenet/watertight_simplified_off/'. Then run script:
```
# render depth map and normal map
python tools/sketch_render/gen_dn_from_mesh.py
# render sketch
python tools/sketch_render/gen_sketch_from_dn.py
# generate boundary sampling
python tools/boundary_sampling.py
```

## Training and Evaluation
Please download our predtrained models from [OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/jzhou67-c_my_cityu_edu_hk/EaSDjDXb4zFKlmBnm64ntZUBKLeaLFmzbyED7jGcTuf_Bw?e=1R0LVv).
```
# Training single-view network
python train_single.py -cat airplane

# Evaluating single-view network
python generate_single.py  -cat airplane -checkpoint 200
python tools/evaluate.py -cat airplane -generation_path experiments/GASv64_airplane/evaluation_200@128_1v/generation/

# Training multi-view network
python train_multi.py -cat airplane

# Evaluating multi-view network
python generate_multi.py  -cat airplane -checkpoint 200 -n 2
python tools/evaluate.py -cat airplane -generation_path experiments/GASv64_airplane/evaluation_200@128_2v/generation/
python generate_multi.py  -cat airplane -checkpoint 200 -n 3
python tools/evaluate.py -cat airplane -generation_path experiments/GASv64_airplane/evaluation_200@128_3v/generation/
```

## User Interface
```
python GA-Sketching-UI/main.py
```
The generated sketches and meshes will be saved in 'cache/'.

