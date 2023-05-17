# MAGDiff: Data Set Shift Detection on Deep Learning Models.
## Readme on how to run the experiments.

You may find the python requirements in the file `requirements.txt`.

### Please follow these instructions to run the experiments:

The pre-trained model-weights `MNIST`, `FMNIST`, `CIFAR10` and `SVHN` are provided in the directory `models`.
The pre-trained model weights for `Imagenette` are directly loaded from the `torchvision` package.
In order to run experiments for the `Imagenette` dataset you will first need to download the dataset from https://github.com/fastai/imagenette and save if in the directory `/data/imagenette`.

Proceed with the following steps:

1) Run the python script `MAGDiff_feature_generation.py` with an input parameter combination from the following parameter grid:
```
	{
	-idn : [MNIST, FMNIST, CIFAR10, SVHN]
	-wgt : [dense3, dense2_1, dense2, fc]
	-sd  : [1.0, 0.5, 0.25]
	-sfn : [gaussian_noise, gaussian_blur, image_shift]
	-sfi : [I, II, III, IV, V, VI]
	}
```

- The `-wgt` parameter controls which layer of the model is considered. For `SVHN` and `CIFAR10`, only `fc` can be used.
	For `MNIST` and `FMNIST`: `dense3` corresponds to `l_-1`, `dense2_1` to `l_-2` and `dense2` to `l_-3` in the notation of the paper.
- The `-sd` parameter controls the `delta`.
- The `-sfn` parameter corresponds to the shift function name.
- The `-sfi` parameter controls the shift intensity. (If you want to generate plots later on you must run each parameter combination for *all* shift intensities.)

Please first run this script for all parameters that you are interested in, for example:
 `python MAGDiff_feature_generation.py -idn MNIST -wgt dense3 -sd 1.0 -sfn gaussian_noise -sfi VI`
 
 This will save the resulting MAGDiff features in the directory `/results/MAGDiff_features`.
 
 2) Next, run the script `MAGDiff_evaluation.py` with parameters from the following parameter grid (only the ones for which you've already completed the previous step):
 ```
	{
	-idn : [MNIST, FMNIST, CIFAR10, SVHN]
	-wgt : [dense3, dense2_1, dense2, fc]
	-sd  : [1.0, 0.5, 0.25]
	-sfn : [gaussian_noise, gaussian_blur, image_shift, ko-shift]
	}
```
This will execute the statistical tests and save the results in the directory `/results/tables`.

3) Once this is done, run the script `collecting_results.py`. This will collect all previously generated results in single files which will be saved in `/results/tables/collected_results/`.

4) Finally, run the script `MAGDiff_plots.py`. This will create the plots, as in the paper, of all the previously generated results. the plots will be saved in the directory `results/figures`.

Note: In the tables, the `PV-BL` entries corresponds to the results for the baseline, called `CV` in the paper, and `MN` corresponds to the MAGDiff matrix norm.
 
