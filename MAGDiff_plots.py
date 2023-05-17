import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from utils.utils_data_analysis import filter_data
from utils.utils_MAGDiff_plots import plot_pivot, plot_accuracies
from sklearn.model_selection import ParameterGrid

pd.options.display.max_columns = None
plot_powers = True

on_cluster = False

curr_dir = os.getcwd()
load_dir = os.path.join(curr_dir, 'results')

figures_base_dir = os.path.join(load_dir, 'figures')
load_dir = os.path.join(load_dir, 'MAGDiff_features')
figures_acc_base_dir = os.path.join(figures_base_dir, 'accuracies')



shift_intensities = ['I', 'II', 'III', 'IV', 'V', 'VI']
deltas = [0.25, 0.5, 1.0]

collected_results_dir = os.path.join(curr_dir, 'results', 'tables', 'collected_results')
collected_powers_file = os.path.join(
    collected_results_dir, 'collected_results_powers.parquet'
)
collected_accuracies_file = os.path.join(
    collected_results_dir, 'collected_model_accuracies.parquet'
)

if plot_powers:
    collected_powers = pd.read_parquet(collected_powers_file)
    param_grid_dict = {
        'data_name': ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN', 'Imagenette'],
        'layer': [('act2', 'dense3'), ('actd2', 'dense2_1'), ('actd1', 'dense2'), ('avgpool', 'fc')],
        'shift_function_name': ['gaussian_noise', 'gaussian_blur', 'image_shift'],
        'modality': ['MN'],
        'delta': [0.25, 1.0, 0.5],
    }
    param_grid = ParameterGrid(param_grid_dict)

    for params in tqdm(param_grid):
        data_name = params['data_name']
        shift_function_name = params['shift_function_name']
        delta = params['delta']
        modality_string = params['modality']
        test_name = 'BBSD'
        weight = params['layer'][1]

        subsample_sizes = [10, 20, 50, 100, 200, 500, 1000]
        if shift_function_name == 'ko_shift' and delta != 1.0:
            continue
        if shift_function_name == 'ko_shift':
            subsample_sizes = [100, 200, 500, 1000]
        if data_name in ['MNIST', 'FMNIST']:
            if weight == 'fc':
                continue
            arc = 'CNN_lightning3'
        else:
            if weight != 'fc':
                continue
            arc = 'ResNet18'

        figures_dir = os.path.join(
            figures_base_dir,
            data_name,
            shift_function_name,
            f'delta_{delta}',
            weight,
            f'{modality_string}',
        )
        Path(figures_dir).mkdir(parents=True, exist_ok=True)

        ## plot w.r.t. shift intensity:
        filter_dict = {
            'reordered': [False],
            'shift_function_name': [shift_function_name],
            'data_name': [data_name],
            'modality': ['PV-BL', modality_string],
            'layer_name': [weight],
            'delta': [delta],
        }

        filtered_powers = filter_data(collected_powers, filter_dict)
        if (list(filtered_powers.modality.unique()) != ['PV-BL', modality_string]):
            print(f"Missing values for parameters:\n {params}\n Skipping!")
            continue

        powers_pivoted_SI = filtered_powers.pivot(
            index='shift_intensity',
            columns=['modality', 'shifted_or_clean', 'subsample_size'],
            values=['power', 'err_-', 'err_+']
        )
        for subsample_size in subsample_sizes:
            plot_pivot(
                powers_pivoted=powers_pivoted_SI,
                fixed_value=subsample_size,
                shift_function_name=shift_function_name,
                figures_dir=figures_dir,
                data_name=data_name,
                shift_delta=delta,
                modality=modality_string,
                alpha=0.05,
                reorder_str='',
                test_name='BBSD',
                savefig=True,
            )

        powers_pivoted_SS = filtered_powers.pivot(
            index='subsample_size',
            columns=['modality', 'shifted_or_clean', 'shift_intensity'],
            values=['power', 'err_-', 'err_+']
        )

        for shift_intensity in shift_intensities:
            plot_pivot(
                powers_pivoted=powers_pivoted_SS,
                fixed_value=shift_intensity,
                shift_function_name=shift_function_name,
                figures_dir=figures_dir,
                data_name=data_name,
                shift_delta=delta,
                modality=modality_string,
                alpha=0.05,
                reorder_str='',
                test_name='BBSD',
                savefig=True,
            )

else:
    collected_accuracies = pd.read_parquet(collected_accuracies_file)
    for mc in ['accuracy_shifted', 'accuracy_clean']:
        collected_accuracies[mc] = collected_accuracies[mc] * 100 #This is to get the accuracy in %.


    param_grid_dict = {
        'data_name': ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN'],
        'shift_function_name': ['gaussian_noise', 'gaussian_blur', 'image_shift', 'ko_shift'],
    }

    param_grid = ParameterGrid(param_grid_dict)

    for params in tqdm(param_grid):
        figures_acc_dir = os.path.join(figures_acc_base_dir, params['data_name'])
        Path(figures_acc_dir).mkdir(parents=True, exist_ok=True)

        filter_dict = {
            'shift_function_name': [params['shift_function_name']],
            'data_name': [params['data_name']],
            'shift_delta': deltas,
        }

        filtered_accuracies = filter_data(collected_accuracies, filter_dict)
        accuracies_pivoted = filtered_accuracies.pivot(
            index=['shift_delta', 'shift_intensity'],
            columns=['data_name'],
            # values=['power', 'err_-', 'err_+']
        )
        if accuracies_pivoted.empty:
            print('No data available for current parameters, please create data first. Skipping!')
            continue
        if params['shift_function_name'] == 'ko_shift':
            plot_deltas = [1.0]
        else:
            plot_deltas = sorted(list(filtered_accuracies['shift_delta'].unique()))

        plot_accuracies(
            accuracies_pivoted,
            params['shift_function_name'],
            figures_acc_dir,
            params['data_name'],
            deltas=plot_deltas,
            savefig=True,
        )

