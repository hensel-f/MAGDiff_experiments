## In this file, the shift functions which are applied to the data are defined.

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torchvision.transforms import GaussianBlur
from keras.preprocessing.image import ImageDataGenerator


### New individual shift function parameters: #####

## Gaussian Blur: #########################
shift_function_params_gaussian_blur_MNIST = {
    'I': {'sigma': 0.35, 'kernel_size': (3, 3)},
    'II': {'sigma': 0.4, 'kernel_size': (3, 3)},
    'III': {'sigma': 0.5, 'kernel_size': (3, 3)},
    'IV': {'sigma': 0.6, 'kernel_size': (3, 3)},
    'V': {'sigma': 0.7, 'kernel_size': (3, 3)},
    'VI': {'sigma': 0.8, 'kernel_size': (3, 3)},

}



shift_function_params_gaussian_blur_SVHN = {
    'I': {'sigma': 2.5, 'kernel_size': (9, 9)},
    'II': {'sigma': 3.0, 'kernel_size': (9, 11)},
    'III': {'sigma': 3.5, 'kernel_size': (11, 11)},
    'IV': {'sigma': 4.0, 'kernel_size': (11, 13)},
    'V': {'sigma': 4.5, 'kernel_size': (13, 13)},
    'VI': {'sigma': 5.0, 'kernel_size': (15, 15)},
    }

shift_function_params_gaussian_blur_CIFAR10 = {
    'I': {'sigma': 1.0, 'kernel_size': (3, 3)},
    'II': {'sigma': 2.0, 'kernel_size': (3, 5)},
    'III': {'sigma': 3.0, 'kernel_size': (5, 5)},
    'IV': {'sigma': 4.0, 'kernel_size': (5, 7)},
    'V': {'sigma': 5.0, 'kernel_size': (7, 7)},
    'VI': {'sigma': 6.0, 'kernel_size': (9, 9)},
    }

shift_function_params_gaussian_blur_FMNIST = shift_function_params_gaussian_blur_MNIST
##############################################

### Gaussian noise: ##########################
shift_function_params_gaussian_noise_MNIST = {
    'I': 25. / 255,
    'II': 40. / 255,
    'III' : 55. / 255,
    'IV' : 70. / 255,
    'V': 85. / 255,
    'VI': 100. / 255
    }

shift_function_params_gaussian_noise_FMNIST = {
    'I': 10. / 255,
    'II': 20. / 255,
    'III': 30. / 255,
    'IV': 40. / 255,
    'V': 50. / 255,
    'VI': 60. / 255,
}

shift_function_params_gaussian_noise_CIFAR10 = {
    'I'  : 30. / 255,
    'II' : 60. / 255,
    'III': 85. / 255,
    'IV' : 100. / 255,
    'V'  : 120. / 255,
    'VI' : 140. / 255,
    }

shift_function_params_gaussian_noise_Imagenette = {
    'I'  : 1. / 255,
    'II' : 8. / 255,
    'III': 25. / 255,
    'IV' : 65. / 255,
    'V'  : 110. / 255,
    'VI' : 170. / 255,
    }

shift_function_params_gaussian_noise_SVHN = shift_function_params_gaussian_noise_CIFAR10
##############################################

### Image shift: #############################
shift_function_params_image_shift_MNIST = {
    'I':
        {
            'rotation_range': 1,
            'width_shift_range': 0.01,
            'height_shift_range': 0.01,
            'shear_range': 0,
            'zoom_range': 0,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'II':
        {
            'rotation_range': 5,
            'width_shift_range': 0.05,
            'height_shift_range': 0.05,
            'shear_range': 0.01,
            'zoom_range': 0.01,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'III':
        {
            'rotation_range': 7,
            'width_shift_range': 0.075,
            'height_shift_range': 0.075,
            'shear_range': 0.02,
            'zoom_range': 0.02,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'IV':
        {
            'rotation_range': 10,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.04,
            'zoom_range': 0.04,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'V':
        {
            'rotation_range': 12.5,
            'width_shift_range': 0.12,
            'height_shift_range': 0.12,
            'shear_range': 0.06,
            'zoom_range': 0.06,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'VI':
        {
            'rotation_range': 25,
            'width_shift_range': 0.15,
            'height_shift_range': 0.15,
            'shear_range': 0.12,
            'zoom_range': 0.12,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
}

shift_function_params_image_shift_FMNIST = {
    'I':
        {
            'rotation_range': 1,
            'width_shift_range': 0.01,
            'height_shift_range': 0.01,
            'shear_range': 0,
            'zoom_range': 0,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'II':
        {
            'rotation_range': 5,
            'width_shift_range': 0.05,
            'height_shift_range': 0.05,
            'shear_range': 0.01,
            'zoom_range': 0.01,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'III':
        {
            'rotation_range': 7,
            'width_shift_range': 0.075,
            'height_shift_range': 0.075,
            'shear_range': 0.02,
            'zoom_range': 0.02,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'IV':
        {
            'rotation_range': 10,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.04,
            'zoom_range': 0.04,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'V':
        {
            'rotation_range': 12.5,
            'width_shift_range': 0.12,
            'height_shift_range': 0.12,
            'shear_range': 0.06,
            'zoom_range': 0.06,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'VI':
        {
            'rotation_range': 15,
            'width_shift_range': 0.15,
            'height_shift_range': 0.15,
            'shear_range': 0.08,
            'zoom_range': 0.08,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
}
shift_function_params_image_shift_CIFAR10 = {

    'I':
        {
            'rotation_range': 20,
            'width_shift_range': 0.175,
            'height_shift_range': 0.175,
            'shear_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'II':
        {
            'rotation_range': 30,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.12,
            'zoom_range': 0.12,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'III':
        {
            'rotation_range': 35,
            'width_shift_range': 0.225,
            'height_shift_range': 0.225,
            'shear_range': 0.15,
            'zoom_range': 0.15,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'IV':
        {
            'rotation_range': 40,
            'width_shift_range': 0.25,
            'height_shift_range': 0.25,
            'shear_range': 0.18,
            'zoom_range': 0.18,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'V':
        {
            'rotation_range': 45,
            'width_shift_range': 0.31,
            'height_shift_range': 0.31,
            'shear_range': 0.23,
            'zoom_range': 0.23,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'VI':
        {
            'rotation_range': 50,
            'width_shift_range': 0.375,
            'height_shift_range': 0.375,
            'shear_range': 0.27,
            'zoom_range': 0.27,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
}

shift_function_params_image_shift_Imagenette = {
    'I':
        {
            'rotation_range': 15,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.08,
            'zoom_range': 0.08,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'II':
        {
            'rotation_range': 20,
            'width_shift_range': 0.14,
            'height_shift_range': 0.14,
            'shear_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'III':
        {
            'rotation_range': 30,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.14,
            'zoom_range': 0.14,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'IV':
        {
            'rotation_range': 40,
            'width_shift_range': 0.25,
            'height_shift_range': 0.25,
            'shear_range': 0.17,
            'zoom_range': 0.17,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'V':
        {
            'rotation_range': 45,
            'width_shift_range': 0.31,
            'height_shift_range': 0.31,
            'shear_range': 0.22,
            'zoom_range': 0.22,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'VI':
        {
            'rotation_range': 60,
            'width_shift_range': 0.37,
            'height_shift_range': 0.37,
            'shear_range': 0.26,
            'zoom_range': 0.26,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
}

shift_function_params_image_shift_SVHN = {
    'I':
        {
            'rotation_range': 20,
            'width_shift_range': 0.15,
            'height_shift_range': 0.15,
            'shear_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'II':
        {
            'rotation_range': 25,
            'width_shift_range': 0.175,
            'height_shift_range': 0.175,
            'shear_range': 0.12,
            'zoom_range': 0.12,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'III':
        {
            'rotation_range': 30,
            'width_shift_range': 0.21,
            'height_shift_range': 0.21,
            'shear_range': 0.15,
            'zoom_range': 0.15,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'IV':
        {
            'rotation_range': 35,
            'width_shift_range': 0.24,
            'height_shift_range': 0.24,
            'shear_range': 0.19,
            'zoom_range': 0.19,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'V':
        {
            'rotation_range': 40,
            'width_shift_range': 0.275,
            'height_shift_range': 0.275,
            'shear_range': 0.21,
            'zoom_range': 0.21,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'VI':
        {
            'rotation_range': 45,
            'width_shift_range': 0.3,
            'height_shift_range': 0.3,
            'shear_range': 0.25,
            'zoom_range': 0.25,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
}

##############################################
##############################################

####ORIGINAL PARAMETERS ####

shift_function_params_gaussian_blur = {
    'I': {'sigma': 1.0, 'kernel_size': (3, 3)},
    'II': {'sigma': 1.0, 'kernel_size': (5, 5)},
    'III': {'sigma': 3.0, 'kernel_size': (5, 5)},
    'IV': {'sigma': 2.0, 'kernel_size': (7, 7)},
    'V': {'sigma': 3.0, 'kernel_size': (7, 7)},
    'VI': {'sigma': 3.0, 'kernel_size': (9, 9)}
    }

shift_function_params_gaussian_noise = {'I'  : 30. / 255,
                                        'II' : 60. / 255,
                                        'III': 85. / 255,
                                        'IV' : 100. / 255,
                                        'V'  : 120. / 255,
                                        'VI' : 140. / 255,
                                        }

shift_function_params_ko_shift = {'I': {'shift_delta': 0.3, 'ko_class': 0},
                                     'II': {'shift_delta': 0.5, 'ko_class': 0},
                                     'III': {'shift_delta': 0.7, 'ko_class': 0},
                                     'IV': {'shift_delta': 0.8, 'ko_class': 0},
                                     'V' : {'shift_delta': 0.9, 'ko_class': 0},
                                     'VI': {'shift_delta': 1.0, 'ko_class': 0},
                                     }

shift_function_params_image_shift = {
    'I':
        {
            'rotation_range': 1,
            'width_shift_range': 0.01,
            'height_shift_range': 0.01,
            'shear_range': 0,
            'zoom_range': 0,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'II':
        {
            'rotation_range': 5,
            'width_shift_range': 0.05,
            'height_shift_range': 0.05,
            'shear_range': 0.01,
            'zoom_range': 0.01,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'III':
        {
            'rotation_range': 7,
            'width_shift_range': 0.075,
            'height_shift_range': 0.075,
            'shear_range': 0.02,
            'zoom_range': 0.02,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'IV':
        {
            'rotation_range': 10,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.04,
            'zoom_range': 0.04,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'V':
        {
            'rotation_range': 12.5,
            'width_shift_range': 0.12,
            'height_shift_range': 0.12,
            'shear_range': 0.06,
            'zoom_range': 0.06,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    'VI':
        {
            'rotation_range': 25,
            'width_shift_range': 0.15,
            'height_shift_range': 0.15,
            'shear_range': 0.12,
            'zoom_range': 0.12,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
}
####################

shift_function_params_gaussian_blur_dict = {
    'MNIST': shift_function_params_gaussian_blur_MNIST,
    'KMNIST': shift_function_params_gaussian_blur_MNIST,
    'FMNIST': shift_function_params_gaussian_blur_FMNIST,
    'CIFAR10': shift_function_params_gaussian_blur_CIFAR10,
    'SVHN': shift_function_params_gaussian_blur_SVHN,
    'Imagenette': shift_function_params_gaussian_blur_SVHN,
}
shift_function_params_gaussian_noise_dict = {
    'MNIST': shift_function_params_gaussian_noise_MNIST,
    'KMNIST': shift_function_params_gaussian_noise_MNIST,
    'FMNIST': shift_function_params_gaussian_noise_FMNIST,
    'CIFAR10': shift_function_params_gaussian_noise_CIFAR10,
    'SVHN': shift_function_params_gaussian_noise_SVHN,
    'Imagenette': shift_function_params_gaussian_noise_Imagenette,
}
shift_function_params_image_shift_dict = {
    'MNIST': shift_function_params_image_shift_MNIST,
    'KMNIST': shift_function_params_image_shift_MNIST,
    'FMNIST': shift_function_params_image_shift_FMNIST,
    'CIFAR10': shift_function_params_image_shift_CIFAR10,
    'SVHN': shift_function_params_image_shift_SVHN,
    'Imagenette': shift_function_params_image_shift_Imagenette,
}


def image_shift(input_data, params=shift_function_params_image_shift['I'], intensity='I', not_batched=False):
    # assert(intensity in ['I', 'II', 'III', 'IV', 'V', 'VI'])
    if isinstance(input_data, np.ndarray):
        input_data = torch.from_numpy(input_data)
    if len(input_data.shape) == 2:
        input_data = torch.unsqueeze(input_data, 0)
    if not_batched == False:
        if len(input_data.shape) == 3:
            input_data = torch.unsqueeze(input_data, 1)
        batch_size = input_data.shape[0]
    else:
        if len(input_data.shape) == 3:
            input_data = torch.unsqueeze(input_data, 0)
        batch_size = 1
    params.update({
        'fill_mode': 'nearest',
        'data_format': 'channels_first',
        })
    datagen = ImageDataGenerator(**params)#(**params[intensity])
    output = datagen.flow(input_data, batch_size=batch_size, shuffle=False)[0]
    if batch_size == 1 and not_batched:
        output = np.squeeze(output)
    return torch.from_numpy(output)

def gaussian_noise(input_data, std, normalization=1.0, clip=False):
    if isinstance(input_data, np.ndarray):
        input_data = torch.from_numpy(input_data)
    shifted_data = torch.normal(input_data, std / normalization)
    if clip:
        return torch.clamp(shifted_data, 0., 1.)
    else:
        return shifted_data

def gaussian_blur(input_data, sigma=1.0, kernel_size=(3,3), unsqueeze=True):
    if isinstance(input_data, np.ndarray):
        input_data = torch.from_numpy(input_data)
    if len(input_data.shape) < 3:
        input_data = torch.unsqueeze(input_data, 0)
        if unsqueeze:
            input_data = torch.unsqueeze(input_data, 0)
    elif len(input_data.shape) < 4 and unsqueeze:
        input_data = torch.unsqueeze(input_data, 1)
    GB = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    res = GB(input_data)
    return res

def random_pixel(x, n):
    res = []
    for x_i in x:
        new_x_i = x_i.copy()
        pixels = [np.random.randint(len(x_i)) for _ in range(n)]
        for j in pixels:
            new_x_i[j] = 1 - new_x_i[j]
        res.append(new_x_i)
    return np.array(res)

def random_pixel_orig_shape(x, n):
    k = x.shape[1]
    assert(k == x.shape[2])
    x_copy = x.copy()
    for x_i in x_copy:
        idx = [(np.random.randint(k), np.random.randint(k)) for _ in range(n)]
        for i,j in idx:
            x_i[i,j] = 1 - x_i[i,j]
    return x_copy

def interpolate_to_gaussian(x, t):
    g = np.random.randn(x.shape[0], x.shape[1])
    return (1-t) * x + t * g
