### Ubuntu 22.04 Preparation (cu12.1)
```bash
apt update
apt install xvfb libosmesa6-dev libgl1-mesa-glx libglfw3 libglx-mesa0 libgl1-mesa-dri

mkdir -p /usr/lib/dri
SWRAST_PATH=$(find /usr -name 'swrast_dri.so' 2>/dev/null)
ln -sf "$SWRAST_PATH" /usr/lib/dri/swrast_dri.so
```
#### Check if cuda is installed
```bash
nvcc --version
```

If not installed, install cuda 12.1
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
chmod +x cuda_12.1.0_530.30.02_linux.run
./cuda_12.1.0_530.30.02_linux.run --override
```

### Environment Setup
```bash
conda create -n safepo python=3.8
conda activate safepo

conda install -c conda-forge gcc

pip install -e ./isaac-gym/python
pip install -e ./safe-policy-optimization
pip install -e ./safety-gymnasium
```

### Example Usage
#### Parameter Tuning
Modify the `train_scripts/marl_cfg/YOUR_ALGO/config.yaml` file to tune the hyperparameters.

#### Commands
```bash
cd datagen

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
python train_scripts/macpo.py --task FreightFrankaCloseDrawer --experiment benchmark
```