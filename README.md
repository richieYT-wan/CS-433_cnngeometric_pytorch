# Machine Learning - Project 2: Advanced deep learning techniques for C. Elegans brain images

**Authors:**

- Richie Yat-tsai Wan
- Pedro Bedmar-López
- Kevin Longlai Qiu

## Running the program 

To run `run.py`, check the options and docstring.
Example of run : 
```
python -W ignore run.py --datapath '../worm_data/video/frames/' --datatype image --alignment-target '../worm_data/video/frames/frame_508.jpg'
```

By default, is uses our best custom architecture for Feature Extraction and Feature Regression.

If datatype video or h5 is specified, it will first extract the frames from the files and then align the frames to either a specified target.

For those h5 or video, please specify the target as frame_number, 
ex : If you know the frame number, specify 'frame_162' as target. 

When using a video, give an estimate frame number based on video FPS and duration.

------------------

To run `demo.ipynb`, download the base weights and extract them under `./trained_models/base_weights/`

[base_Affine Weights](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_streetview_checkpoint_adam_affine_grid_loss_PAMI.pth.tar)

[base_TPS Weights](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_streetview_checkpoint_adam_tps_grid_loss_PAMI.pth.tar)

Weights for Resnet101 are too large to fit on Github. You can download them [here](https://drive.google.com/file/d/1KN13tTdfkbroPnjwoFOK1lnL9X4wY9AT/view?usp=sharing) instead.

To get the useable .h5 data, download it [here](https://drive.google.com/file/d/13EA_BLrneB1tDyk0leslCsU6eZagJH1D/view?usp=sharing) and save it as `more_data.h5` under `./worm_data/h5_data/`

To get the old .h5 data, download it [here](https://drive.google.com/file/d/15yliIRD1o6EifZ4v4zn-Zklfw_NIlEpL/view?usp=sharing) and save it as `epfl3.h5` under `./worm_data/h5_data/`

For the video, download it [here](https://drive.google.com/file/d/1AFeFE7KMB8yIGI9T-sdZ7pZS9bMoZ0ln) and save it as `
20200624_142043_crop_0_2_4.avi` under `./worm_data/video/`

## Folder structure
```
├── project2
    ├── code
    |   └── `data/` : scripts to download the datasets used by Rocco et al.
    |   └── `datasets/` : contains the training/validation datasets
    |   └── `geotnf/` : methods for applying geometric transformations
    |   └── `image/` : methods for image handling such as normalization
    |   └── `model/` : .py files for model creation.
    |   └── `options/` : options
    |   └── `trained_models/` : trained weights saved under here
    |   └── `util/` : various utility methods.
    |  *notebooks are to be ran in the following order.*
    |   └── `0_data_extraction.ipynb`
    |   └── `1_data_generation.ipynb`
    |   └── `2_apply_model.ipynb`
    |   └── `demo.ipynb`
    |   └── `eval.py`
    |   └── `train.py`
    └── worm_data
        └── `h5_data/` : contains the .h5 files and all associated files created from it.
        └── `video/` : contains the .avi video and all associated files created from it.

```
## Description of scripts/notebooks ...
