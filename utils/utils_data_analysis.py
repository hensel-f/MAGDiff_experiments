## Some helper functions related to the data collection.
import os

def get_tables_load_dir(params, base_dir):
    '''
    Returns the directory to load experimental results of specific parameter values.
    '''
    data_name = params['data_name']
    shift_function_name = params['shift_function_name']
    shift_delta = params['shift_delta']
    layer_names_weight = params['layer'][-1]
    load_dir = os.path.join(base_dir, data_name, shift_function_name, f'delta_{shift_delta}',
                 layer_names_weight)
    return load_dir

def check_files_exist(files_list):
    '''
    Checks if all files in the input list exist.
    '''
    for f in files_list:
        if not os.path.isfile(f):
            return False
    return True

def valid_parameters(params):
    '''
    Helper function that checks if a given dictionary containing experiment parameters is "valid".
    '''
    if params['shift_function_name'] == 'ko_shift' and params['shift_delta'] != 1.0:
        return False
    elif params['arch'] == 'ResNet18':
        if (not params['layer'] in [('avgpool', 'fc')]):
            return False
        if not params['data_name'] in ['CIFAR10', 'SVHN', 'Imagenette']:
            return False
        return True
    if params['arch'] == 'CNN_lightning3':
        if (not params['layer'] in [('act2', 'dense3'), ('actd2', 'dense2_1'), ('actd1', 'dense2'), ('flatten', 'dense1')]):
            return False
        if not params['data_name'] in ['MNIST', 'FMNIST']:
            return False
        return True
    else:
        return True


def cross_columns(dataframe, cross_columns, drop_original=False, separator='__'):
    new_col_name = separator.join(cross_columns)
    dataframe[new_col_name] = dataframe[cross_columns[0]].astype(str)
    for cross_col in cross_columns[1:]:
        dataframe[new_col_name] += separator + dataframe[cross_col].astype(str)

    if drop_original:
        dataframe.drop(columns=cross_columns)
    return dataframe

def filter_data(dataframe, filter_dict):
    df = dataframe.copy()
    for key, val in filter_dict.items():
        df = df[dataframe[key].isin(val)]
    return df
