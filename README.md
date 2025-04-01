# <center>OffsetOPT: Explicit Surface Reconstruction without Normals [<a href="https://arxiv.org/abs/2503.15763" style="color:#1e90ff;">arXiv preprint</a>](https://arxiv.org/abs/2503.15763)</center>


![alt text](Method.png)

**OffsetOPT** is a method for explicit surface reconstruction from 3D point clouds **without relying on point normals**.  It introduces a two-stage framework:  (1) a neural network trained to predict local surface triangles from clean, uniformly sampled point clouds, and  (2) a per-point offset optimization applied to new input point clouds — including both small-scale shapes and large-scale scenes — for surface reconstruction.  The method is effective for accurate surface reconstruction and sharp detail preservation.

---

## 🔧 Installation
- All experiments in this project were conducted using the following configuration:
  - GPU: NVIDIA RTX 4090  
  - CUDA: 12.1  
  - PyTorch: 2.3.0

- We recommend using [**mamba**](https://mamba.readthedocs.io/en/latest/) for fast and reliable environment setup:

  ```bash
  # Install mamba, create the environment, and activate it
  conda install -y mamba -n base -c conda-forge
  mamba env create -f environment.yml -y
  conda activate OffsetOPT 
  ```
- All required data can be obtained by running `bash download_data.sh`.
  It will download and store all data in the subfolder ./Data of the project directory.

## **Stage 1: Train the triangulation network**
  - We provide pretrained models:
    - `knn=50`: `trained_models/model_knn50.pth`
    - `knn=100`: `trained_models/model_knn100.pth`
  - To train the model from your own side, run:
    ```bash
    python S1_train.py  # the default setting is knn=50
    ```

## **Stage 2: Optimize point offsets to reconstruct surfaces**
> By default, the provided `model_knn50.pth` is used.
  - Run the following command for point clouds formed by GT mesh vertices:
    ```bash
    python S2_reconstruct.py --delta=0.0 --dataset=ABC_test  # replace 'ABC_test' with `FAUST`, or `MGN` for the respective datasets
    ```

  - Run the following command for dense point clouds:
    ```bash
    python S2_reconstruct.py --delta=0.02 --dataset=ScanNet
    python S2_reconstruct.py --delta=0.1 --dataset=Matterport3D
    python S2_reconstruct.py --delta=0.1 --dataset=CARLA_1M
    python S2_reconstruct.py --delta=0.01 --dataset=Thingi10k --rescale_delta
    python S2_reconstruct.py --delta=0.015 --dataset=Stanford3D --rescale_delta
    ```
    - For **Matterport3D**, there are 6 scenes (listed in `MP3D_5cm.lst`) in relatively small scale (<15 meters).  
      We reconstruct them using a finer resolution (`--delta=0.05`, i.e., 5 cm) rather than 10 cm in the released results.

    - The reconstructed meshes of **Stanford3D** scan data [[link](https://graphics.stanford.edu/data/3Dscanrep/)] contain bumpy vertices, which can be smoothed using one iteration of Laplacian smoothing using `python smooth_Stanford3D.py`.  The results with and without smoothing are stored in smoothed_Stanford3D/ and Stanford3D/, respectively.


## **Evaluation of Reconstructed Surfaces**
> You can download our reconstructed meshes from using `bash download_results.sh`.
  - shape evaluation (sample 100K points): 
    ```bash
    # ABC: 6771 meshes — this will take time
    python main_eval_acc.py --gt_path=./Data/ABC/test --pred_path=./results/ABC_test

    # FAUST:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/FAUST --pred_path=./results/FAUST

    # MGN:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/MGN --pred_path=./results/MGN

    # Thingi10K:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/Thingi10K --pred_path=./results/Thingi10K

    # Stanford3D:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/Stanford3D --pred_path=./results/Stanford3D
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/Stanford3D --pred_path=./results/smoothed_Stanford3D
    ```

  - scene evaluation (sample 1 Million points): 
    ```bash
    # ScanNet:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/ScanNet --pred_path=./results/ScanNet --sample_num=1000000 

    # Matterport3D:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/Matterport3D --pred_path=./results/Matterport3D --sample_num=1000000 

    # CARLA:
    python main_eval_acc.py --gt_path=./Data/GT_Meshes/CARLA --pred_path=./results/CARLA_1M --sample_num=1000000 
    ```


### Citation
If you use this work, please cite:
```
@article{lei2025offsetopt,
  title={OffsetOPT: Explicit Surface Reconstruction without Normals},
  author={Lei, Huan},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```