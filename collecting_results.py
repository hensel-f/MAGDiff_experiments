## Collects the result dataframes obtained and saved by "MAGDiff_evaluation.py" and saves them in a file.

import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import pandas as pd
from sklearn.model_selection import ParameterGrid
import utils.utils_data_analysis as uda

param_grid_dict = {
    'arch'                          : ['CNN_lightning3', 'ResNet18'],
    'matrix_norm_only'              : [True],
    'add_only_PV'                   : [True, False],
    'data_name'                     : ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN', 'Imagenette'],
    'layer'                         : [('act2', 'dense3'), ('actd2', 'dense2_1'), ('actd1', 'dense2'), ('avgpool', 'fc')],
    'shift_function_name'           : ['gaussian_noise', 'ko_shift', 'gaussian_blur', 'image_shift'],
    'shift_delta'                   : [1.0, 0.5, 0.25],
}

param_grid = ParameterGrid(param_grid_dict)

on_cluster = False

overwrite = True
collect_all_tables = True
collect_model_accuracies = True

curr_dir = os.getcwd()
load_dir = os.path.join(curr_dir, 'results', 'tables')
load_dir_accuracies = os.path.join(curr_dir, 'results', 'MAGDiff_features')


save_dir = os.path.join(load_dir, 'collected_results')
Path(save_dir).mkdir(parents=True, exist_ok=True)

collected_results_p_vals_file = os.path.join(save_dir, 'collected_results_p_vals.parquet')
collected_results_powers_file = os.path.join(save_dir, 'collected_results_powers.parquet')
collected_results_powers_file_csv = os.path.join(save_dir, 'collected_results_powers.csv')
collected_model_accuracies_file = os.path.join(save_dir, 'collected_model_accuracies.parquet')


if not overwrite:
    files_list = [
        collected_results_powers_file_csv,
        collected_results_powers_file,
        collected_results_p_vals_file,
        collected_model_accuracies_file
    ]
    files_exist = []
    for fi in files_list:
        files_exist.append(os.path.isfile(fi))
    if all(files_exist):
        print(f'Terminating. Overwrite is set to {overwrite} and the files already exist.')
        quit()


delete_existing_files = True
if delete_existing_files and collect_all_tables:
    files_list = [collected_results_powers_file_csv, collected_results_powers_file, collected_results_p_vals_file]
    for fi in files_list:
        if os.path.isfile(fi):
            os.remove(fi)

collected_p_vals = None
collected_powers = None
row_accuracies = {}
row_accuracies_df = []

for params in tqdm(param_grid):
    if not uda.valid_parameters(params):
        ## Check if the parameters are "valid".
        continue

    add_only_PV = params['add_only_PV']
    if add_only_PV:
        p_vals_f_name = 'p_vals_PV.parquet'
        powers_f_name = 'powers_PV.parquet'
    else:
        p_vals_f_name = 'p_values.parquet'
        powers_f_name = 'powers.parquet'

    modality = 'MN'
    modality_string = modality

    if collect_model_accuracies and params['matrix_norm_only'] and params['add_only_PV']:
        if (params['arch'] == 'CNN_lightning3' and params['layer'] == ('act2', 'dense3')) or (params['arch'] == 'ResNet18' and params['layer'] == ('avgpool', 'fc')):

            for intensity in ['I', 'II', 'III', 'IV', 'V', 'VI']:
                features_dir = os.path.join(
                    load_dir_accuracies,
                    params['data_name'],
                    params['shift_function_name'],
                    f'delta_{params["shift_delta"]}',
                    params['layer'][1],
                    f'{modality_string}')
                clean_TDA_dict_file = os.path.join(
                    features_dir,
                    f'clean_{modality_string}_dict' + '.pkl'
                )
                shifted_TDA_dict_file = os.path.join(
                    features_dir,
                    f'shifted_{modality_string}_dict_{params["shift_function_name"]}_{intensity}' + '.pkl'
                )
                try:
                    with open(clean_TDA_dict_file, 'rb') as handle:
                        TDA_features_dic_keep = pkl.load(handle)
                    with open(shifted_TDA_dict_file, 'rb') as handle:
                        TDA_features_dic_shift_keep = pkl.load(handle)
                except:
                    print(f'The file: {shifted_TDA_dict_file}\n does not exist. Please create it first. Skipping!')
                    continue

                model_test_acc = TDA_features_dic_keep.get('test_acc', 'NA')
                model_test_acc_shifted = TDA_features_dic_shift_keep.get('test_acc_shifted', 'NA')
                if model_test_acc != 'NA':
                    model_test_acc = round(model_test_acc[0]['test_acc'], 3)
                    model_test_acc_shifted = round(model_test_acc_shifted, 3)

                row_accuracies.update({
                    'data_name': params['data_name'],
                    'shift_function_name': params['shift_function_name'],
                    'shift_intensity': intensity,
                    'accuracy_clean': model_test_acc,
                    'accuracy_shifted': model_test_acc_shifted,
                    'shift_delta': params['shift_delta'],
                })
                row_accuracies_df.append(row_accuracies.copy())

    if collect_all_tables:
        param_load_dir = uda.get_tables_load_dir(params, load_dir)
        if not add_only_PV:
            param_load_dir = os.path.join(param_load_dir, f'{modality_string}')

        p_vals_file = os.path.join(param_load_dir, p_vals_f_name)
        powers_file = os.path.join(param_load_dir, powers_f_name)

        file_list = [p_vals_file, powers_file]

        if not uda.check_files_exist(file_list):
            print(f'Skipping! Files are missing at:\n {param_load_dir}')
            continue

        if not isinstance(collected_p_vals, pd.DataFrame):
            collected_p_vals = pd.read_parquet(p_vals_file)
        else:
            params_p_vals = pd.read_parquet(p_vals_file)
            collected_p_vals = pd.concat([collected_p_vals, params_p_vals], ignore_index=True)

        if not isinstance(collected_powers, pd.DataFrame):
            collected_powers = pd.read_parquet(powers_file)
        else:
            params_powers = pd.read_parquet(powers_file)
            collected_powers = pd.concat([collected_powers, params_powers], ignore_index=True)

## Save the collected results:
if collect_all_tables:
    collected_powers.to_parquet(collected_results_powers_file, compression=None)
    collected_p_vals.to_parquet(collected_results_p_vals_file, compression=None)
    collected_powers.to_csv(collected_results_powers_file_csv)
if collect_model_accuracies:
    accuracies_df = pd.DataFrame(row_accuracies_df)
    accuracies_df.to_parquet(collected_model_accuracies_file, compression=None)

print(f'Collected results saved at:\n {save_dir}')
