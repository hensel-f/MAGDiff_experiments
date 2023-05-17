import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

shift_name_dict = {
    'gaussian_noise': 'Gaussian noise',
    'image_shift': 'Image shift',
    'gaussian_blur': 'Gaussian blur',
}

def plot_pivot(
    powers_pivoted,
    fixed_value,
    shift_function_name,
    figures_dir,
    data_name,
    shift_delta,
    modality='MN',
    alpha=0.05,
    reorder_str='',
    test_name='BBSD',
    savefig=True,
    ):
        modality_str = modality
        baseline_str = 'PV-BL'
        plot_wrt = powers_pivoted.index.name
        sub_dir_name = plot_wrt
        xlabel_name =plot_wrt.replace('_', ' ')
        
        assert(plot_wrt in ['shift_intensity', 'subsample_size'])
        if plot_wrt == 'shift_intensity':
            fixed_value_name = 'subsample size'
        else:
            fixed_value_name = 'shift intensity'

        pivot_level = [0, 1, -1]
        pivot_value = fixed_value

        linewidth = 1.5
        capsize = 2.5
        markersize = 4.0
        title_fontsize = 18
        axes_fontsize = 16
        ticks_fontsize = 14
        legend_fontsize = 16
        color_alpha = 0.3

        col_rename_dict_uerrs = {'clean': 'clean_uerrs', 'shifted': 'shifted_uerrs'}

        ## preparation of data:
        powers_CV = powers_pivoted.xs(
            ('power', baseline_str, pivot_value),
            axis=1,
            level=pivot_level,
        )
        lerrs_CV = powers_pivoted.xs(
            ('err_-', baseline_str, pivot_value),
            axis=1,
            level=pivot_level,
        )
        uerrs_CV = powers_pivoted.xs(
            ('err_+', baseline_str, pivot_value),
            axis=1,
            level=pivot_level,
        )

        CV_data = powers_CV.join(lerrs_CV, lsuffix='_powers', rsuffix='_lerrs', sort=True)
        CV_data = CV_data.join(uerrs_CV.rename(columns=col_rename_dict_uerrs), rsuffix='_uerrs', sort=True)

        powers_modality = powers_pivoted.xs(
            ('power', modality_str, pivot_value),
            axis=1,
            level=pivot_level
        )
        lerrs_modality = powers_pivoted.xs(
            ('err_-', modality_str, pivot_value),
            axis=1,
            level=pivot_level
        )
        uerrs_modality = powers_pivoted.xs(
            ('err_+', modality_str, pivot_value),
            axis=1,
            level=pivot_level
        )
        
        modality_data = powers_modality.join(lerrs_modality, lsuffix='_powers', rsuffix='_lerrs', sort=True)
        modality_data = modality_data.join(uerrs_modality.rename(columns=col_rename_dict_uerrs), rsuffix='_uerrs', sort=True)

        
        x_values = CV_data.index.values
        assert((x_values == modality_data.index.values).all())
        plt.figure(1, figsize=(7.5, 6))
        plt.clf()

        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)

        plt.ylim([-0.1, 1.1])
        if plot_wrt == 'subsample_size':
            plt.xscale('log', base=10)

        plt.xlabel(xlabel_name, fontsize=axes_fontsize)
        plt.ylabel('power estimation', fontsize=axes_fontsize)
        plt.axhline(y=alpha, color='black', linestyle='--', linewidth=1.7)

        plt.plot(
            x_values,
            CV_data['shifted_powers'],
            ':', color='limegreen', marker='o',
            linewidth=linewidth,
            markersize=markersize,
            label='power CV'
        )
        plt.errorbar(x_values,
                     CV_data['shifted_powers'],
                     np.vstack((CV_data['shifted_lerrs'], CV_data['shifted_uerrs'])),
                     color='limegreen', fmt='o', capsize=capsize
                     )
        plt.fill_between(
            x_values,
            CV_data['shifted_powers'] - CV_data['shifted_lerrs'],
            CV_data['shifted_powers'] + CV_data['shifted_uerrs'],
            alpha=color_alpha,
            color='limegreen',
        )

        plt.plot(
            x_values,
            modality_data['shifted_powers'],
            ':', color='red', marker='o',
            linewidth=linewidth,
            markersize=markersize,
            label=f'power {modality_str}'
        )
        plt.errorbar(x_values,
                     modality_data['shifted_powers'],
                     np.vstack((modality_data['shifted_lerrs'], modality_data['shifted_uerrs'])),
                     color='red', fmt='o', capsize=capsize
                     )
        plt.fill_between(
            x_values,
            modality_data['shifted_powers'] - modality_data['shifted_lerrs'],
            modality_data['shifted_powers'] + modality_data['shifted_uerrs'],
            alpha=color_alpha,
            color='red',
        )

        plt.plot(
            x_values,
            CV_data['clean_powers'],
            ':', color='darkgreen', marker='x',
            linewidth=linewidth,
            markersize=markersize,
            label='Type I error CV'
        )
        plt.errorbar(x_values,
                     CV_data['clean_powers'],
                     np.vstack((CV_data['clean_lerrs'], CV_data['clean_uerrs'])),
                     color='darkgreen', fmt='x', capsize=capsize
                     )
        plt.fill_between(
            x_values,
            CV_data['clean_powers'] - CV_data['clean_lerrs'],
            CV_data['clean_powers'] + CV_data['clean_uerrs'],
            alpha=color_alpha,
            color='darkgreen',
        )
        
        plt.plot(
            x_values,
            modality_data['clean_powers'],
            ':', color='brown', marker='x',
            linewidth=linewidth,
            markersize=markersize,
            label=f'Type I error {modality_str}'
        )
        plt.errorbar(x_values,
                     modality_data['clean_powers'],
                     np.vstack((modality_data['clean_lerrs'], modality_data['clean_uerrs'])),
                     color='brown', fmt='x', capsize=capsize
                     )
        plt.fill_between(
            x_values,
            modality_data['clean_powers'] - modality_data['clean_lerrs'],
            modality_data['clean_powers'] + modality_data['clean_uerrs'],
            alpha=color_alpha,
            color='brown',
            )
        plt.grid()
        plt.legend(prop={'size': legend_fontsize})

        plt.title(
            f'{data_name}: {shift_name_dict[shift_function_name]}, {fixed_value_name}: {fixed_value}',
            fontsize=title_fontsize,
        )

        if savefig:
            Path(os.path.join(figures_dir, sub_dir_name)).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(figures_dir, sub_dir_name,
                                     f'data_shift_power_{test_name}_{reorder_str}_{shift_function_name}_{fixed_value}' +
                                     '.png'), bbox_inches='tight', dpi=200)

        plt.close('all')


def plot_accuracies(
        accuracies_pivoted,
        shift_function_name,
        figures_dir,
        data_name,
        deltas=[0.25, 0.5, 1.0],
        savefig=True,
):
    fixed_value_name = 'shift intensity'
    xlabel_name = fixed_value_name

    pivot_level = 0

    linewidth = 2.0
    capsize = 2.5
    markersize = 6.0
    title_fontsize = 18
    axes_fontsize = 16
    ticks_fontsize = 14
    legend_fontsize = 16
    color_alpha = 0.8

    colors = ['red', 'green', 'blue', 'orange']

    plt.figure(1, figsize=(7.5, 6))
    plt.clf()

    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    plt.xlabel(xlabel_name, fontsize=axes_fontsize)
    plt.ylabel('accuracy (%)', fontsize=axes_fontsize)

    for idx, delta in enumerate(deltas):
        acc_delta = accuracies_pivoted.xs(
            (delta),
            axis=0,
            level=pivot_level,
        )

        x_values = acc_delta.index.values
        y_values = acc_delta['accuracy_shifted'][data_name]

        plt.plot(
            x_values,
            y_values,
            ':',
            color=colors[idx],
            marker='o',
            linewidth=linewidth,
            markersize=markersize,
            label=f'{delta}',
            alpha=color_alpha,
        )

    plt.grid()
    plt.legend(prop={'size': legend_fontsize})

    plt.title(
        f'{data_name}: model accuracies for {shift_name_dict[shift_function_name]}',
        fontsize=title_fontsize,
    )

    if savefig:
        Path(figures_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(
            os.path.join(figures_dir, f'model_accuracy_{data_name}_{shift_function_name}' + '.png'),
            bbox_inches='tight',
            dpi=200,
        )

    plt.close('all')