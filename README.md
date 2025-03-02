# Pairing Matters

Code for the paper [On Semantic Misalignments in Image-Text Pairing for Multimodal Contrastive Learning](https://example.com). 

This repo contains code for numerical and semi-controlled simulations (see './Simulation') and experiments on real-world data (see './RealworldExp'). 

## 1 Simulation
```bash
# CD to directory at first
$ cd ./Simulation
```

### Installation
```bash
# install dependencies (preferably, inside your conda/virtual environment)
$ pip install -r requirements.txt

# test if pytorch was installed with cuda support; should not raise an error
$ python -c "import torch; assert torch.cuda.device_count() > 0, 'No cuda support'"
```

