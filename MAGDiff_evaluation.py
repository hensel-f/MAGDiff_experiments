import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import torch
import utils.utils_shift_functions as usf
import utils.utils_shift_detection as usd

parser = argparse.ArgumentParser()

parser.add_argument(
    '-sfn',
    '--shift_function_name',
    type=str,
    required=False,
    default='gaussian_noise',
    help='Name of the shift, currently supported: "gaussian_noise", "gaussian_blur", "ko_shift" or "image_shift".',
)
parser.add_argument(
    '-sfi',
    '--shift_function_intensity',
    type=str,
    nargs='+',
    required=False,
    default=["I", "II", "III", "IV", "V", "VI"],
    help='List of intensities of the shift, currently supported: ["I", "II", "III", "IV", "V", "VI"].',
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
    '-o',
    '--overwrite',
    required=False,
    action='store_true',
    help='Control whether to re-run the statistical test and overwrite previous results.',
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

overwrite = args.overwrite
on_cluster = False
use_cuda = False

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
print('Num threads: ', torch.get_num_threads(), '\nNum CPUs: ', os.cpu_count())

curr_dir = os.getcwd()
load_dir = os.path.join(curr_dir, 'results', 'MAGDiff_features')

tables_base_dir = os.path.join(curr_dir, 'results', 'tables')
Path(tables_base_dir).mkdir(parents=True, exist_ok=True)

data_name = args.in_distribution_data
arc = 'CNN_lightning3'
if data_name in ['CIFAR10','SVHN', 'Imagenette']:
    arc = 'ResNet18'
test_name = 'BBSD'
weights = args.weights
num_TD_layers = len(weights)


##Setting the function for the statistical test:
shift_function_name = args.shift_function_name
shift_delta = args.shift_delta
if shift_function_name == 'image_shift':
    shift_function = usf.image_shift
    shift_function_params = usf.shift_function_params_image_shift
elif shift_function_name == 'gaussian_noise':
    shift_function = usf.gaussian_noise
    shift_function_params = usf.shift_function_params_gaussian_noise
elif shift_function_name == 'gaussian_blur':
    shift_function = usf.gaussian_blur
    shift_function_params = usf.shift_function_params_gaussian_blur
else:
    NotImplemented('Please choose "image_shift", "gaussian_noise", "ko_shift" or "gaussian_blur" as a shift function.')

shift_function_intensities = args.shift_function_intensity

sample_with_replacement = True

max_num_tests = 1500
subsample_sizes = [10, 20, 50, 100, 200, 500, 1000]

test_function = usd.BBSD_from_pred
test_function_kwargs = {}

matrix_norm_flag = True

num_classes = 10


matrix_norm_string = 'matrix_norm'
modality = 'MN'
modality_string = 'TD'

layer_names_list = [{'weight': weights[i]} for i in range(num_TD_layers)]

for layer_names in layer_names_list:
    rows_p_values_df = []
    rows_powers_df = []
    rows_p_values_df_PV = []
    rows_powers_df_PV = []

    features_dir = os.path.join(load_dir, data_name, shift_function_name, f'delta_{shift_delta}', layer_names['weight'],
                                f'{modality}')
    Path(features_dir).mkdir(parents=True, exist_ok=True)

    tables_dir_PV = os.path.join(tables_base_dir, data_name, shift_function_name, f'delta_{shift_delta}',
                                 layer_names['weight'],
                                 )
    tables_dir = os.path.join(tables_dir_PV, f'{modality}')
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    powers_PV_file = os.path.join(tables_dir_PV, 'powers_PV.parquet')
    p_vals_PV_file = os.path.join(tables_dir_PV, 'p_vals_PV.parquet')

    create_PV_table = True


    if arc in ['Net_lightning', 'Net_lightning3']:
        checkpoints = ['weights-epoch=9.ckpt', 'weights-epoch=0.ckpt']
    elif arc == 'ResNet18':
        if data_name == 'CIFAR10':
            checkpoints = [
                'weights-epoch=49.ckpt']
        elif data_name == 'SVHN':
            checkpoints = [
                'weights-epoch=20.ckpt']
        elif data_name in ['Imagenette']:
            checkpoints = ['ResNet18_Weights.IMAGENET1K_V1']
    else:
        checkpoints = [
                       'weights-epoch=49.ckpt']

    ##################################################################
    ## Loading all TD/TU (resp. matrix norm) features of the model
    ## for the unperturbed as well as the shifted training data and
    ## executing the statistical tests:
    ##################################################################
    powers = {}
    lerrs = {}
    uerrs = {}
    errs = {}
    model_test_accuracies_shifted = {}

    alpha = 0.05

    reorder_TDs = False
    reorder_str = ''
    if reorder_TDs:
        reorder_str = 'reordered'

    power_estimations_features_dir = os.path.join(features_dir, f'power_estimations')
    Path(power_estimations_features_dir).mkdir(parents=True, exist_ok=True)
    powers_save_file = os.path.join(power_estimations_features_dir,
                                    f'powers_{reorder_str}'+'.pkl')
    errs_save_file = os.path.join(power_estimations_features_dir,
                                  f'errs_{reorder_str}' +'.pkl')
    subsample_sizes_save_file = os.path.join(power_estimations_features_dir,
                                  f'subsample_sizes_{reorder_str}' + '.pkl')

    try:
        assert(overwrite == False)
        with open(powers_save_file, 'rb') as handle:
            powers = pkl.load(handle)
        with open(errs_save_file, 'rb') as handle:
            errs = pkl.load(handle)
        with open(subsample_sizes_save_file, 'rb') as handle:
            subsample_sizes_loaded = pkl.load(handle)
            assert(subsample_sizes == subsample_sizes_loaded)
            subsample_sizes = subsample_sizes_loaded
            del subsample_sizes_loaded
    except:
        for intensity in shift_function_intensities:

            clean_TDA_dict_file = os.path.join(features_dir, f'clean_{modality}_dict' +  '.pkl')
            shifted_TDA_dict_file = os.path.join(features_dir, f'shifted_{modality}_dict_{shift_function_name}_{intensity}' +'.pkl')

            with open(clean_TDA_dict_file, 'rb') as handle:
                TDA_features_dic_keep = pkl.load(handle)
            with open(shifted_TDA_dict_file, 'rb') as handle:
                TDA_features_dic_shift_keep = pkl.load(handle)

            powers.update({intensity: {f'{test_name}_pred_clean': [], f'{test_name}_pred_shift': [], f'{test_name}_TDA_clean': [], f'{test_name}_TDA_shift': [], 'combined_clean': [], 'combined_shift': []}})
            lerrs.update({intensity: {f'{test_name}_pred_clean': [], f'{test_name}_pred_shift': [], f'{test_name}_TDA_clean': [], f'{test_name}_TDA_shift': [], 'combined_clean': [], 'combined_shift': []}})
            uerrs.update({intensity: {f'{test_name}_pred_clean': [], f'{test_name}_pred_shift': [], f'{test_name}_TDA_clean': [], f'{test_name}_TDA_shift': [], 'combined_clean': [], 'combined_shift': []}})
            errs.update({intensity: {}})

            for subsample_size in tqdm(subsample_sizes):
                num_tests = max_num_tests

                TDA_features_clean = TDA_features_dic_keep[f'{modality_string}_all']
                predicted_probas_clean = TDA_features_dic_keep['predicted_probas']
                TDA_features_shift = TDA_features_dic_shift_keep[f'{modality_string}_all']
                predicted_probas_shift = TDA_features_dic_shift_keep['predicted_probas']
                # if reorder_TDs:
                #     TDA_features_clean = usd.reorder_wrt_predictions(predicted_probas_clean, TDA_features_clean)
                #     TDA_features_shift = usd.reorder_wrt_predictions(predicted_probas_shift, TDA_features_shift)

                test_TDA_features_shifted = []
                test_TDA_features_clean = []
                test_pred_features_shifted = []
                test_pred_features_clean = []
                combined_clean = []
                combined_shifted = []

                for t in range(num_tests):
                    total_num_clean = TDA_features_clean.shape[0]
                    total_num_shifted = TDA_features_shift.shape[0]
                    if shift_function_name != 'ko_shift':
                        assert(total_num_clean == predicted_probas_clean.shape[0] == total_num_shifted  == predicted_probas_shift.shape[0])
                    subsample_mask = np.random.choice(np.arange(total_num_clean), subsample_size, replace=sample_with_replacement)
                    subsample_mask_2 = np.random.choice(np.arange(total_num_clean), subsample_size, replace=sample_with_replacement)
                    subsample_mask_shifted = np.random.choice(np.arange(total_num_shifted), subsample_size, replace=sample_with_replacement)

                    TD_clean_subsamp = TDA_features_clean[subsample_mask]
                    predicted_probas_clean_subsamp = predicted_probas_clean[subsample_mask]
                    combined_clean_subsamp = np.concatenate((predicted_probas_clean_subsamp, TD_clean_subsamp))

                    TD_clean_subsamp_2 = TDA_features_clean[subsample_mask_2]
                    predicted_probas_clean_subsamp_2 = predicted_probas_clean[subsample_mask_2]
                    combined_clean_subsamp_2 = np.concatenate((predicted_probas_clean_subsamp_2, TD_clean_subsamp_2))

                    TD_shift_subsamp = TDA_features_shift[subsample_mask_shifted]
                    predicted_probas_shift_subsamp = predicted_probas_shift[subsample_mask_shifted]
                    combined_shift_subsamp = np.concatenate((predicted_probas_shift_subsamp, TD_shift_subsamp))


                    decisions_TDA_features_clean, p_vals_TDA_features_clean = test_function(TD_clean_subsamp, TD_clean_subsamp_2, alpha=alpha, **test_function_kwargs)
                    # append row to dataframes:
                    row_p_vals_df = {'data_name': data_name, 'test_name': test_name, 'shift_function_name': shift_function_name,
                                     'delta': shift_delta, 'architecture': arc, 'layer_name': layer_names['weight'],
                                     'modality': modality,
                                     'shift_intensity': intensity, 'subsample_size': subsample_size,
                                     'p_value': p_vals_TDA_features_clean, 'shifted_or_clean': 'clean' , 'reordered': reorder_TDs}
                    rows_p_values_df.append(row_p_vals_df.copy())

                    test_TDA_features_clean.append(decisions_TDA_features_clean)


                    decisions_pred_features_clean, p_vals_pred_features_clean = test_function(predicted_probas_clean_subsamp, predicted_probas_clean_subsamp_2, alpha=alpha, **test_function_kwargs)
                    if create_PV_table:
                        row_p_vals_df.update({'modality': 'PV-BL', 'p_value': p_vals_pred_features_clean})
                        # PV-BL stands for 'prediction (probabilities) vector - baseline'.
                        rows_p_values_df_PV.append(row_p_vals_df.copy())

                    test_pred_features_clean.append(decisions_pred_features_clean)

                    ## combined will not be added to the tables.
                    combined_clean.append(test_function(combined_clean_subsamp, combined_clean_subsamp_2, alpha=alpha, **test_function_kwargs)[0])


                    decisions_TDA_features_shifted, p_vals_TDA_features_shifted = test_function(TD_clean_subsamp, TD_shift_subsamp, alpha=alpha, **test_function_kwargs)
                    row_p_vals_df.update({'modality': modality, 'p_value': p_vals_TDA_features_shifted, 'shifted_or_clean': 'shifted'})
                    rows_p_values_df.append(row_p_vals_df.copy())

                    test_TDA_features_shifted.append(decisions_TDA_features_shifted)

                    decisions_pred_features_shifted, p_vals_pred_features_shifted = test_function(predicted_probas_clean_subsamp, predicted_probas_shift_subsamp, alpha=alpha, **test_function_kwargs)
                    if create_PV_table:
                        row_p_vals_df.update({'modality': 'PV-BL', 'p_value': p_vals_pred_features_shifted})
                        rows_p_values_df_PV.append(row_p_vals_df.copy())
                    test_pred_features_shifted.append(decisions_pred_features_shifted)
                    combined_shifted.append(test_function(combined_clean_subsamp, combined_shift_subsamp, alpha=alpha, **test_function_kwargs)[0])



                est, lerr, uerr = usd.tests_conf_interval(np.array(test_pred_features_clean), alpha=alpha)
                powers[intensity][f'{test_name}_pred_clean'].append(est), lerrs[intensity][f'{test_name}_pred_clean'].append(lerr), uerrs[intensity][f'{test_name}_pred_clean'].append(uerr)

                row_powers_df = {'data_name': data_name, 'test_name': test_name, 'shift_function_name': shift_function_name,
                                     'delta': shift_delta, 'architecture': arc, 'layer_name': layer_names['weight'],
                                     'modality': 'PV-BL',
                    'shift_intensity': intensity, 'subsample_size': subsample_size, 'power': est, 'err_-': lerr, 'err_+': uerr , 'shifted_or_clean': 'clean', 'reordered': reorder_TDs}
                if create_PV_table:
                    rows_powers_df_PV.append(row_powers_df.copy())

                est, lerr, uerr = usd.tests_conf_interval(np.array(test_pred_features_shifted), alpha=alpha)
                powers[intensity][f'{test_name}_pred_shift'].append(est), lerrs[intensity][
                    f'{test_name}_pred_shift'].append(lerr), uerrs[intensity][f'{test_name}_pred_shift'].append(
                    uerr)
                row_powers_df.update({'power': est, 'err_-': lerr, 'err_+': uerr, 'shifted_or_clean': 'shifted'})
                if create_PV_table:
                    rows_powers_df_PV.append(row_powers_df.copy())


                est, lerr, uerr = usd.tests_conf_interval(np.array(test_TDA_features_clean), alpha=alpha)
                powers[intensity][f'{test_name}_TDA_clean'].append(est), lerrs[intensity][f'{test_name}_TDA_clean'].append(lerr), uerrs[intensity][f'{test_name}_TDA_clean'].append(uerr)
                row_powers_df.update({'power': est, 'err_-': lerr, 'err_+': uerr, 'shifted_or_clean': 'clean', 'modality': modality})
                rows_powers_df.append(row_powers_df.copy())
                est, lerr, uerr = usd.tests_conf_interval(np.array(test_TDA_features_shifted), alpha=alpha)
                powers[intensity][f'{test_name}_TDA_shift'].append(est), lerrs[intensity][f'{test_name}_TDA_shift'].append(lerr), uerrs[intensity][f'{test_name}_TDA_shift'].append(uerr)
                row_powers_df.update({'power': est, 'err_-': lerr, 'err_+': uerr, 'shifted_or_clean': 'shifted', 'modality': modality})
                rows_powers_df.append(row_powers_df.copy())

                est, lerr, uerr = usd.tests_conf_interval(np.array(combined_clean), alpha=alpha)
                powers[intensity]['combined_clean'].append(est), lerrs[intensity]['combined_clean'].append(lerr), uerrs[intensity]['combined_clean'].append(uerr)
                est, lerr, uerr = usd.tests_conf_interval(np.array(combined_shifted), alpha=alpha)
                powers[intensity]['combined_shift'].append(est), lerrs[intensity]['combined_shift'].append(lerr), uerrs[intensity]['combined_shift'].append(
                    uerr)


                #print(powers)
            for k in [f'{test_name}_pred_clean', f'{test_name}_pred_shift', f'{test_name}_TDA_clean', f'{test_name}_TDA_shift', 'combined_clean', 'combined_shift']:
                errs[intensity][k] = np.concatenate((lerrs[intensity][k], uerrs[intensity][k])).reshape((2, len(subsample_sizes)))


powers_df = pd.DataFrame(rows_powers_df)
p_vals_df = pd.DataFrame(rows_p_values_df)

powers_df.to_parquet(os.path.join(tables_dir, 'powers.parquet'), compression=None)
p_vals_df.to_parquet(os.path.join(tables_dir, 'p_values.parquet'), compression=None)
powers_df.to_csv(os.path.join(tables_dir, 'powers.csv'))

if create_PV_table:
    powers_df_PV = pd.DataFrame(rows_powers_df_PV)
    p_vals_df_PV = pd.DataFrame(rows_p_values_df_PV)
    powers_df_PV.to_parquet(powers_PV_file, compression=None)
    p_vals_df_PV.to_parquet(p_vals_PV_file, compression=None)
    powers_df_PV.to_csv(os.path.splitext(powers_PV_file)[0] + '.csv')
