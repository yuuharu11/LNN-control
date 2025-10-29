# Environment Setup Guide for LTC Training with Robomimic/Robosuite

## Quick Start
### 1. Install MiniConda
```bash
apt install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source $HOME/.bashrc && conda --version
```

### 2. Create Conda Environment

```bash
conda create -n robomimic_venv python=3.10
conda activate robomimic_venv
```

### 3. Install Dependencies
```bash
# Install robosuite
pip install robosuite

# Install system libraries
apt update
apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# For EGL (GPU rendering - recommended if /dev/dri exists)
apt install -y libegl1-mesa libegl1-mesa-dev libgles2-mesa-dev

# For OSMesa (CPU rendering - fallback option)
apt install -y libosmesa6-dev
```

### 3.1 Fix libstdc++ compatibility for OSMesa rendering
```bash
# Option 1: Update conda's libstdc++ (Recommended)
conda install -c conda-forge libstdcxx-ng

# Option 2: Prioritize system libraries (Alternative)
# Add to ~/.bashrc:
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. Install Robomimic and Robosuite from Source

```bash
# Install robomimic
cd /work/robosuite/robomimic
pip install -e .

# Install robosuite
cd /work/robosuite
pip install -e .
```

### 5. Verify Installation

```bash
# Test PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Test robosuite
python -c "import robosuite; print('Robosuite version:', robosuite.__version__)"

# Test ncps
python -c "import ncps; print('NCP/LTC imported successfully')"
```

## Alternative: Full Installation

If you want all dependencies (including optional packages):

```bash
pip install -r requirements_full.txt
```

## Troubleshooting

### Issue: ModuleNotFoundError for robosuite

**Solution:**
```bash
export PYTHONPATH="/work/robosuite:$PYTHONPATH"
```

Or add to your `.bashrc`:
```bash
echo 'export PYTHONPATH="/work/robosuite:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

### Issue: CUDA version mismatch

**Solution:**
Check your CUDA version:
```bash
nvidia-smi
```

Then install the matching PyTorch version from https://pytorch.org/get-started/locally/

### Issue: EGL/OSMesa rendering errors

**Solution 1: Use EGL with GPU (Recommended - Fast)**
```bash
# Verify DRI devices are available
ls -la /dev/dri/  # Should show renderD128, card0, etc.

# Set environment variables for EGL
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Add to ~/.bashrc for persistence
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
echo 'export PYOPENGL_PLATFORM=egl' >> ~/.bashrc
source ~/.bashrc

# Verify
python -c "import os; os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'; import mujoco; print('✅ EGL OK - GPU rendering enabled')"
```

**Solution 2: Use OSMesa (Software Rendering - Slower but works anywhere)**
```bash
# Install OSMesa
apt install -y libosmesa6-dev

# Update libstdc++ for compatibility
conda install -c conda-forge libstdcxx-ng

# Set environment variables
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Verify
python -c "import os; os.environ['MUJOCO_GL']='osmesa'; os.environ['PYOPENGL_PLATFORM']='osmesa'; import mujoco; print('✅ OSMesa OK - CPU rendering')"
```

**Solution 3: No rendering mode (for training without camera obs)**
```python
env = suite.make(
    "Lift",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
```

**Rendering Backend Comparison:**
| Backend | Speed | GPU | Requirements |
|---------|-------|-----|--------------|
| EGL | ⚡⚡⚡ Fast | ✅ Yes | `/dev/dri` devices |
| OSMesa | 🐌 Slow | ❌ CPU only | `libosmesa6-dev` |
| No render | ⚡⚡⚡ Fastest | N/A | None (no visual obs) |

### Issue: Protobuf version conflicts

**Solution:**
Force install the correct version:
```bash
pip install protobuf==3.20.3 --force-reinstall
```

## Package Version Summary

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10.x | Base environment |
| PyTorch | >=2.0.0 | Deep learning framework |
| pytorch-lightning | >=2.0.0 | Training framework |
| ncps | >=1.0.0 | LTC/NCP neural networks |
| tensorflow | 2.14.0 | Robomimic compatibility |
| mujoco | >=3.0.0 | Physics simulation |
| robosuite | 1.5.1 | Robotics environments |
| hydra-core | >=1.3.0 | Configuration management |

## Testing Your Setup

### Test 1: Train on UCI-HAR Dataset

```bash
cd /work/liquid_time-constant_networks
python train.py experiment=uci_har/ncp dataset=uci_har train.seed=1
```

### Test 2: Train on Robomimic Dataset

```bash
cd /work/liquid_time-constant_networks
./train_scripts/robomimic/ncp_lift.sh
```

### Test 3: Evaluate with Rollouts (requires robosuite)

```bash
cd /work/liquid_time-constant_networks
./eval_robosuite.sh
```

## Known Issues

1. **torchvision image.so warning**: This is a known issue with pre-built torchvision wheels. It doesn't affect training. To fix, rebuild torchvision from source or ignore the warning.

2. **Pydantic warnings**: UnsupportedFieldAttributeWarning can be safely ignored. These are compatibility warnings between pydantic and other packages.

3. **TF32 precision warnings**: PyTorch 2.9+ changed default precision settings. These warnings can be safely ignored for most use cases.

## Conda Environment Export

To export your exact environment:

```bash
conda activate robomimic_venv
conda env export > environment.yml
pip freeze > requirements_frozen.txt
```

To recreate from export:

```bash
conda env create -f environment.yml
```

## Docker Alternative

If you prefer Docker:

```bash
# Build image
docker build -t ltc-robomimic .

# Run container
docker run --gpus all -it -v /work:/work ltc-robomimic
```

## Additional Resources

- PyTorch Installation: https://pytorch.org/get-started/locally/
- Robosuite Documentation: https://robosuite.ai/
- NCP/LTC Paper: https://www.nature.com/articles/s42256-020-00237-3
- Hydra Documentation: https://hydra.cc/

## License

See individual package licenses for details.

## Citation

If you use this codebase, please cite:

```bibtex
@article{hasani2020liquid,
  title={Liquid time-constant networks},
  author={Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Rus, Daniela and Grosu, Radu},
  journal={arXiv preprint arXiv:2006.04439},
  year={2020}
}

@inproceedings{robosuite2020,
  title={robosuite: A modular simulation framework and benchmark for robot learning},
  author={Zhu, Yuke and Wong, Josiah and Mandlekar, Ajay and Mart{\'i}n-Mart{\'i}n, Roberto and Joshi, Abhishek and Nasiriany, Soroush and Zhu, Yifeng},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```
