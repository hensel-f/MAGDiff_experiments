import os
from pathlib import Path
import argparse
import numpy as np
import pickle as pkl
import torch
import utils.utils_optimization as uo
import utils.utils_architectures as ua
import utils.utils_shift_functions as usf
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import pytorch_lightning as pl
from torchvision.models.resnet import ResNet18_Weights
from utils.utils_imagenette import Imagenette

use_cuda = False
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    '-sfn',
    '--shift_function_name',
    type=str,
    required=False,
    default='gaussian_noise',
    help='Name of the shift, currently supported: "gaussian_noise", "image_shift", "gaussian_blur".',
)
parser.add_argument(
    '-sfi',
    '--shift_function_intensity',
    type=str,
    required=False,
    default='I',
    help='Intensity of the shift, currently supported: "I", "II", "III", "IV", "V", "VI".',
)
parser.add_argument(
    '-sd',
    '--shift_delta',
    type=float,
    required=False,
    default=1.,
    help='Percentage of the dataset the shift will be applied to.',
)

parser.add_argument(
    '-idn',
    '--in_distribution_data',
    type=str,
    required=False,
    default='MNIST',
    help='Name of dataset to be used for training, currently supported: "MNIST", "FMNIST", "CIFAR10".',
)

parser.add_argument(
    '-wgt',
    '--weights',
    type=str,
    nargs='+',
    required=False,
    default=['dense3'],
    help='List of weights which are used to compute the TD.'
         'Must correspond to the correct activations, and match their order!',
)

args = parser.parse_args()

use_cuda = False

overwrite = True

data_dir = os.path.join(os.getcwd(), 'data')
Path(data_dir).mkdir(parents=True, exist_ok=True)
curr_dir = os.getcwd()

data_dir_mnist = os.path.join(data_dir, 'mnist')
data_dir_fmnist = os.path.join(data_dir, 'fmnist')
data_dir_cifar10 = os.path.join(data_dir, 'cifar10')
data_dir_svhn = os.path.join(data_dir, 'svhn')
data_dir_imagenette = os.path.join(data_dir, 'imagenette')

num_classes = 10

matrix_norm_string = 'matrix_norm'
modality_string = 'MN'
modality = 'matrix norm'
matrix_norm_string_ckpt = ''


nopco = 1000

batch_size = 128
lr = 0.001
data_name = args.in_distribution_data

## Getting the shift-function:
shift_function_name = args.shift_function_name
shift_delta = args.shift_delta
if shift_function_name == 'image_shift':
    shift_function = usf.image_shift
    shift_function_params = usf.shift_function_params_image_shift_dict[data_name]
elif shift_function_name == 'gaussian_noise':
    shift_function = usf.gaussian_noise
    shift_function_params = usf.shift_function_params_gaussian_noise_dict[data_name]
elif shift_function_name == 'gaussian_blur':
    shift_function = usf.gaussian_blur
    shift_function_params = usf.shift_function_params_gaussian_blur_dict[data_name]
else:
    NotImplemented('Please choose "image_shift", "gaussian_noise" or "gaussian_blur" as a shift function.')

shift_function_intensities = [args.shift_function_intensity]

weights = args.weights

if weights[0] == 'dense3':
    act = 'act2'
elif weights[0] == 'dense2_1':
    act = 'actd2'
elif weights[0] == 'dense2':
    act = 'actd1'
elif weights[0] == 'fc':
    act = 'avgpool'

layer_names = {'activation': act, 'weight': weights[0]}

## Load architecture:
arc = 'CNN_lightning3'
if data_name in ['CIFAR10', 'SVHN', 'Imagenette']:
    arc = 'ResNet18'
    batch_size = 64
    lr = 0.1

if arc == 'CNN_lightning3':
        net = ua.CNN_lightning3(num_classes=num_classes).to(device)
elif arc == 'ResNet18':
    if not data_name in ['Imagenette']:
        net = ua.ResNet18(num_classes=num_classes).to(device)
    else:
        assert(data_name in ['Imagenette'])
        net = ua.ResNet18_pl().to(device)
else:
    RuntimeError('Chosen architecture not supported!')


## Preparing for plots:
features_save_dir = os.path.join(curr_dir, 'results', 'MAGDiff_features', data_name, shift_function_name, f'delta_{shift_delta}', layer_names['weight'],
                            f'{modality_string}')
Path(features_save_dir).mkdir(parents=True, exist_ok=True)

checkpoint_dir = os.path.join(curr_dir, 'models')


# preparation of training data and test data:
data_name_dict = {'MNIST': MNIST, 'FMNIST': FashionMNIST, 'CIFAR10': CIFAR10, 'SVHN': SVHN}
data_dirs_dict = {'MNIST': data_dir_mnist, 'FMNIST': data_dir_fmnist, 'CIFAR10': data_dir_cifar10, 'SVHN': data_dir_svhn}

if data_name in ['Imagenette']:
    transformation = ResNet18_Weights.IMAGENET1K_V1.transforms()
else:
    transformation = transforms.Compose([transforms.ToTensor()])

if data_name not in ['SVHN', 'Imagenette']:
    train_data = data_name_dict[data_name](data_dirs_dict[data_name], train=True, download=True, transform=transformation)
    test_data = data_name_dict[data_name](data_dirs_dict[data_name], train=False, download=True, transform=transformation)
elif data_name == 'Imagenette':
    train_data = Imagenette(data_dir_imagenette, 'train', transformation)
    test_data = Imagenette(data_dir_imagenette, 'val', transformation)
else:
    train_data = data_name_dict[data_name](data_dirs_dict[data_name], split='train', download=True, transform=transformation)
    test_data = data_name_dict[data_name](data_dirs_dict[data_name], split='test', download=True, transform=transformation)

test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=False,
                         pin_memory=True,
                         )
if data_name in ['Imagenette']:
    train_loader = DataLoader(train_data,
                             batch_size=batch_size,  # ,num_observations,
                             shuffle=True,
                             pin_memory=True,
                             )


## set up the comparison dataloader:
if data_name != 'SVHN':
    train_data_targets = train_data.targets
else:
    train_data_targets = train_data.labels

mpc_sampler = MPerClassSampler(train_data_targets, nopco, nopco * num_classes)

comparison_dataloader = DataLoader(train_data,
                                   batch_size=nopco * num_classes,
                                   sampler=mpc_sampler,
                                   pin_memory=True,
                                   )

# train_data_labels_per_cl = uo.data_per_class_dl(uo.inf_train_gen(comparison_dataloader).__next__(), device=device)
comparison_data_labels_per_class = uo.data_per_class_dl(uo.inf_train_gen(comparison_dataloader).__next__(),
                                                        device=device)
## Loading the weights of the pretrained model:
criterion = torch.nn.CrossEntropyLoss().to(device)
model = uo.TD_optimizer_lightning(net, learning_rate=lr,
                                  loss_criterion=criterion,
                                  ).to(device)

## Getting the observations
test_observations_targets = uo.inf_train_gen(test_loader).__next__()
targets = test_observations_targets[1].to(device)
test_observations = test_observations_targets[0].to(device)


##################################################################
## Computing all features of the loaded model
# for the unperturbed as well as the shifted  data:
##################################################################

for cpt in [f'{data_name}.ckpt']:
    print('checkpoint:', cpt)
    checkpoint = os.path.join(checkpoint_dir, cpt)

    ##Loading the model and checkpoint:
    trainer = pl.Trainer(limit_train_batches=0, limit_val_batches=0, callbacks=[TQDMProgressBar(refresh_rate=500)])
    if data_name not in ['Imagenette']:
        test_acc = trainer.test(model, ckpt_path=checkpoint, dataloaders=test_loader)
    else:
        trainer.fit(model, train_loader, test_loader)
        test_acc = trainer.test(model, dataloaders=test_loader)


    model.to(device)

    mean_features_file = os.path.join(checkpoint_dir,
                                      f'{data_name}_mean_features_act_{act}' + '.pkl')
    try:
        with open(mean_features_file, 'rb') as handle:
            mean_features_per_label_list = pkl.load(handle)
        compute_mean_features_matrices = False
    except:
        compute_mean_features_matrices = True

    for intensity in shift_function_intensities:
        clean_dict_file = os.path.join(features_save_dir, f'clean_{modality_string}_dict' + '.pkl')
        shifted_dict_file = os.path.join(features_save_dir, f'shifted_{modality_string}_dict_{shift_function_name}_{intensity}' + '.pkl')

        try: # Try to load the features if they already exist.
            assert(overwrite == False)
            with open(clean_dict_file, 'rb') as handle:
                TD_features_dic_keep = pkl.load(handle)
            with open(shifted_dict_file, 'rb') as handle:
                TD_features_dic_shift_keep = pkl.load(handle)
            print('The requested features already exist, not executing computation.')

        except: # Compute and save the features if they don't exist yet.
            ## Applying the shift to the test dataset:
            if data_name not in ['Imagenette']:
                test_data_transformed = test_data.data / 255.
            else:
                test_data_transformed = test_data

            if data_name != 'SVHN':
                test_data_targets = test_data.targets
            else:
                test_data_targets = test_data.labels

            if shift_function_name == 'gaussian_noise':
                std_dev_correction = 1.0
                clip = True
                shift_function_args = {'std': shift_function_params[intensity], 'normalization': std_dev_correction,
                                       'clip': clip}
            elif shift_function_name == 'gaussian_blur':
                shift_function_args = shift_function_params[intensity]
            elif shift_function_name == 'image_shift':
                shift_function_args = {'params': shift_function_params[intensity]}
            else:
                NotImplemented('Please choose "image_shift", "gaussian_noise" or "gaussian_blur" as a shift function.')

            if data_name not in ['Imagenette']:
                if isinstance(test_data.data, np.ndarray):
                    test_data_transformed = torch.from_numpy(test_data_transformed)
                    if data_name == 'CIFAR10':
                        test_data_transformed = test_data_transformed.permute(0,3,1,2)
                if isinstance(test_data_targets, (list, np.ndarray)):
                    test_data_targets = torch.tensor(test_data_targets)

                if shift_delta == 1.0:
                    test_data_shifted = shift_function(test_data_transformed, **shift_function_args)
                    if isinstance(test_data_shifted, tuple):
                        test_data_targets = test_data_shifted[1]
                        test_data_shifted = test_data_shifted[0]
                elif 0 < shift_delta < 1:
                    shift_number = int(shift_delta * test_data_transformed.shape[0])
                    shift_indexes = np.random.choice(np.arange(test_data_transformed.shape[0]), shift_number, replace=False)
                    test_data_shifted = test_data_transformed.clone().detach()
                    shifted_data = shift_function(test_data_transformed[shift_indexes], **shift_function_args).type(test_data_shifted.type())
                    if len(test_data_shifted.shape) == len(shifted_data.shape) - 1:
                        test_data_shifted = torch.unsqueeze(test_data_shifted, 1)
                    test_data_shifted[shift_indexes] = shifted_data
                else:
                    raise ValueError(f"The shift delta must be in (0, 1].\n It is currently set to {shift_delta}.")

                if len(test_data_shifted.shape) == 3:
                    test_data_shifted = torch.unsqueeze(test_data_shifted, 1)
                shifted_dataset = TensorDataset(test_data_shifted.float(), test_data_targets)
            else:
                shifted_dataset = Imagenette(data_dir_imagenette, 'val', transform=transformation,
                                             shift=shift_function_name, shift_args=shift_function_args,
                                             delta=shift_delta)

            test_loader_shifted = DataLoader(shifted_dataset,#test_data_shifted,
                                             batch_size=batch_size,  # ,num_observations,
                                             shuffle=False,
                                             pin_memory=True)

            test_acc_shifted = trainer.test(model, dataloaders=test_loader_shifted)[0]['test_acc']


            if compute_mean_features_matrices:

                mean_features_per_label_list = uo.compute_mean_adjacency_matrices2(
                    model.model,
                    comparison_data_labels_per_class,
                    layer_names=layer_names)

                with open(mean_features_file, 'wb') as mf_file:
                    pkl.dump(mean_features_per_label_list, mf_file)

                compute_mean_features_matrices = False

            features_dic_shift = uo.compute_TD_entropy_means_TD_predcl_per_class2(model.model, test_loader_shifted,  #train_loader,
                                                                                  comparison_data_labels_per_class,
                                                                                  layer_names=layer_names,
                                                                                  mean_adjacency_matrices_list=mean_features_per_label_list,
                                                                                  )

            hist_keys = features_dic_shift.keys()

            TD_features_dic_shift_keep = {key: torch.squeeze(features_dic_shift[key]).numpy() for key in hist_keys}
            TD_features_dic_shift_keep['test_acc'] = test_acc
            TD_features_dic_shift_keep['test_acc_shifted'] = test_acc_shifted
            # save the features:
            with open(shifted_dict_file, 'wb') as handle:
                pkl.dump(TD_features_dic_shift_keep, handle)

            if not os.path.isfile(clean_dict_file):
                TD_features_dic = uo.compute_TD_entropy_means_TD_predcl_per_class2(model.model, test_loader,
                                                                                   comparison_data_labels_per_class,
                                                                                   layer_names=layer_names,
                                                                                   mean_adjacency_matrices_list=mean_features_per_label_list
                                                                                   )

                TD_features_dic_keep = {key: torch.squeeze(TD_features_dic[key]).numpy() for key in hist_keys}
                TD_features_dic_keep['test_acc'] = test_acc
                # save the features:
                with open(clean_dict_file, 'wb') as handle:
                    pkl.dump(TD_features_dic_keep, handle)
