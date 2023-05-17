
import time
import numpy as np
from tqdm import tqdm

import torch
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
import torch.optim as optim



class TD_optimizer_lightning(pl.LightningModule):
    def __init__(self, model, learning_rate, loss_criterion):
        super(TD_optimizer_lightning, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_criterion = loss_criterion


    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        loss = self.loss_criterion(output, target)
        # gradient_analysis = False

        preds = torch.argmax(output, dim=1)
        acc = accuracy(preds, target)
        self.log('loss', loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)
        return {'loss': loss, 'acc': acc}

    def evaluate(self, batch, stage=None):
        data, target = batch
        output = self.model(data)
        loss = self.loss_criterion(output, target)
        preds = torch.argmax(output, dim=1)
        acc = accuracy(preds, target)

        #if stage:
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

        return preds, target

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")


    def validation_epoch_end(self, outputs):
        outs = tuple(map(torch.cat, zip(*outputs)))
        all_preds = outs[0]
        all_targets = outs[1]
        acc = accuracy(all_preds, all_targets, task='multiclass', num_classes=10)
        self.log('validation_acc_epoch', acc, on_epoch=True, on_step=False, prog_bar=True)
        return acc

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        if self.model.name == 'ResNet':
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
            return optimizer


def inf_train_gen(dataloader):
    '''
    Function that enables infinitely iterating through a dataloader.
    '''
    while True:
        for images, targets in dataloader:
            yield images, targets
    return


def data_per_class_dl(dataloader_batch, device=torch.device('cpu')):
    '''
    Given a batch of a pytorch Dataloader, returns a list of tensors of all samples in each target class and a sorted
    list of all labels occurring in the batch.

    :param dataloader_batch: batch of a pytorch Dataloader
    :return: a tuple containing a list of tensors of the data part only of the data samples grouped
             by their label, and a sorted list of all labels occurring in the batch.
    '''
    # choose a random batch containing a fixed number of representatives of each label:
    #opt_batch = random.choice(list(dataloader)).to(device)
    opt_batch = dataloader_batch
    opt_batch_sorted = sorted(list(zip(torch.unbind(opt_batch[0], dim=0), torch.unbind(opt_batch[1], dim=0))),
                              key=lambda x: x[1])
    labels_unique = torch.unique(opt_batch[1], sorted=True)

    data_all_classes = []
    for t in labels_unique:
        data_class = torch.stack([d for (d, label) in opt_batch_sorted if label == t]).to(device)
        data_all_classes.append(data_class)

    return (data_all_classes, labels_unique)

def get_activation(dict, detach=True):
    def hook(model, input, output):
        if detach:
            dict['activation'] = output.detach()
        else:
            dict['activation'] = output
    return hook


def activation_matrix_from_nodes_and_weights(act_nodes, weight_matrix):
    '''
    Computes the activation matrix from activation of the previous layer and the following weight matrix.
    :param act_nodes: pytorch tensor of the activation nodes, dim = n-samples x size-of-incoming-layer.
    :param weight_matrix: pytorch tensor containing the weight matrix of the layer,
                          dim = size-left-layer x size-right-layer.
    '''
    act_nodes = torch.squeeze(act_nodes)
    if len(act_nodes.shape) == 1:
        act_nodes = torch.unsqueeze(act_nodes, 0)
    A_diag_ext = torch.stack([torch.diag(a) for a in act_nodes])
    act_matrix = torch.stack([torch.mm(a, torch.t(weight_matrix)) for a in A_diag_ext])
    # using torch.matmul would be preferable here, but it is not compatible with previous torch versions.
    return act_matrix


def p_norm(a, p=2., average=True, no_root=True):
    '''
    :param a: tensor of shape (1,).
    :param p: Exponent of the norm in [1,+inf].
    '''
    normalization_factor = 1.
    if average:
        normalization_factor = a.size(dim=0)

    if p == 2.:
        return torch.div(torch.linalg.norm(a), normalization_factor)
    elif p == np.inf:
        return torch.div(torch.max(a), normalization_factor)
    else:
        r = torch.pow(a, p)
        if not no_root:
            r = torch.pow(r, (1./p))
        return torch.div(r,  normalization_factor)


def compute_TD2(net, samples, comparison_data_labels_per_class,
               aggregation=None, p=2.,
               mean_adjacency_matrices_per_label_list=None, matrix_norm_only=True,
               layer_names={'activation': 'act2', 'weight': 'dense3'}):
    stime = time.time()
    model_name = net.name
    with torch.no_grad():
        predicted_classes = net(samples).argmax(dim=1)
        ### added for hooks:
        # weight matrix:
        weight_activation = {}
        if model_name == 'ResNet':
            model = net.model
            weight_activation['weight'] = getattr(model, layer_names['weight']).weight.detach()#model.fc.weight.detach()
            mean_hook = getattr(model, layer_names['activation']).register_forward_hook(get_activation(weight_activation))
        else:
            weight_activation['weight'] = getattr(net, layer_names['weight']).weight.detach()
            mean_hook = getattr(net, layer_names['activation']).register_forward_hook(get_activation(weight_activation))
        ###

        all_labels = comparison_data_labels_per_class[1]
        comparison_data_classes = comparison_data_labels_per_class[0]

        ## computing the mean adjacency matrices if they are not passed as inputs:
        if mean_adjacency_matrices_per_label_list == None:
            mean_adjacency_matrices_per_label_list = []
            for cdata_cl in comparison_data_classes:
                net(cdata_cl)
                ## ResNet exception:
                if model_name == 'ResNet':
                    weight_activation['activation'] = torch.flatten(weight_activation['activation'], 1)

                waa_mean = torch.mean(weight_activation['activation'], dim=0)
                mean_AM = activation_matrix_from_nodes_and_weights(waa_mean, weight_activation['weight'])
                MAM = torch.squeeze(mean_AM)

                #######
                mean_adjacency_matrices_per_label_list.append(MAM)
            mean_hook.remove()

        mean_adjacency_matrices_per_label = torch.squeeze(torch.stack(mean_adjacency_matrices_per_label_list))

        res = []
        batch_weight_act = {}
        if model_name == 'ResNet':
            batch_weight_act['weight'] = getattr(model,
                                                  layer_names['weight']).weight.detach()
            sample_hook = getattr(model, layer_names['activation']).register_forward_hook(
                get_activation(batch_weight_act))
        else:
            batch_weight_act['weight'] = getattr(net, layer_names['weight']).weight.detach()
            sample_hook = getattr(net, layer_names['activation']).register_forward_hook(
                          get_activation(batch_weight_act))
        net(samples)
        ## ResNet exception:
        if model_name == 'ResNet':
            batch_weight_act['activation'] = torch.flatten(batch_weight_act['activation'], 1)

        act_matrix = activation_matrix_from_nodes_and_weights(batch_weight_act['activation'],
                                                              batch_weight_act['weight'])
        assert(predicted_classes.shape[0] == act_matrix.shape[0])
        matrix_diff_all = torch.stack(
            [torch.stack(
                [torch.abs(act_matrix[i] - mean_adjacency_matrices_per_label[j]) for j in range(mean_adjacency_matrices_per_label.shape[0])]
                        ) for i in range(predicted_classes.shape[0])]
             )

        for i in range(matrix_diff_all.shape[0]):
            if matrix_norm_only:
                res.append(torch.stack([torch.linalg.norm(md, ord='fro') for md in matrix_diff_all[i]]))
        return res


def compute_TD_entropy_means_TD_predcl_per_class2(net, test_data_loader, comparison_data_labels_per_class,
                                                  layer_names={'activation': 'act2', 'weight': 'dense3'},
                                                 mean_adjacency_matrices_list=None,
                                                 num_classes=10,
                                                 matrix_norm_only=True,
                                                 max_samples=np.inf,
                                                 device=torch.device('cpu'),
                                                 TU_computation=False):

    with torch.no_grad():
        try:
            device = net.device
        except:
            device = device

        all_TDs = []
        all_TDs_pred_class = []
        all_TUs = []
        all_TUs_pred_class = []
        all_pred_classes = []
        all_pred_probas = []
        all_targets = []
        # stopper = 0
        curr_num_samples = 0

        for data in tqdm(test_data_loader, miniters=20, disable=False):

            if curr_num_samples >= max_samples:
                break

            samples = data[0].to(device)

            targets = data[1].to(device)
            curr_num_samples += targets.shape[0]

            all_targets.append(targets)
            predicted_probas = net(samples)
            predicted_classes = predicted_probas.argmax(dim=1)
            all_pred_classes.append(predicted_classes)
            all_pred_probas.append(predicted_probas)
            if TU_computation == False:
                all_TDs.append(torch.stack(compute_TD2(net, samples, comparison_data_labels_per_class, layer_names=layer_names,
                       mean_adjacency_matrices_per_label_list=mean_adjacency_matrices_list,
                      matrix_norm_only=matrix_norm_only)))
                all_TDs_pred_class.append(all_TDs[-1][torch.arange(all_TDs[-1].size(0)), predicted_classes])
                
            else:
                all_TUs.append(compute_TU(net, samples, comparison_data_labels_per_class, layer_names=layer_names,
                      all_classes=True, mean_PDs_per_label_list=mean_adjacency_matrices_list,
                      matrix_norm_only=matrix_norm_only))
                all_TUs_pred_class.append(all_TUs[-1][torch.arange(all_TUs[-1].size(0)), predicted_classes])


        if TU_computation == False:
            all_TDs = torch.cat(all_TDs, dim=0)
            all_TDs_pred_class = torch.cat(all_TDs_pred_class, dim=0)
            all_entropies_TD = compute_entropy_torch(all_TDs)
        else:
            all_TUs = torch.cat(all_TUs, dim=0)
            all_TUs_pred_class = torch.cat(all_TUs_pred_class, dim=0)
            all_entropies_TU = compute_entropy_torch(all_TUs)
        all_pred_classes = torch.cat(all_pred_classes, dim=0)
        all_pred_probas = torch.cat(all_pred_probas, dim=0)
        all_targets = torch.cat(all_targets, dim=0)


        dic = {
            'TD_only_pred_class': all_TDs_pred_class.cpu() if not TU_computation else torch.empty(0),
            'TD_all': all_TDs.cpu() if not TU_computation else torch.empty(0),
            'TU_only_pred_class': all_TUs_pred_class.cpu() if TU_computation else torch.empty(0),
            'TU_all': all_TUs.cpu() if TU_computation else torch.empty(0),
            'predicted_probas': all_pred_probas.cpu(),
            'entropies_sum_TD': all_entropies_TD.cpu() if not TU_computation else torch.empty(0),
            'entropies_sum_TU': all_entropies_TU.cpu() if TU_computation else torch.empty(0),
            'predicted_classes': all_pred_classes.cpu(),
            'targets': all_targets.cpu(),
        }
        return dic


def compute_entropy_torch(TD_all_classes):
    '''
    TD_all_classes: input array of n_samples x n_classes
    returns: entropy of softmax ot TD vectors.

    '''
    return torch.sum(torch.special.entr(torch.nn.Softmax(dim=1)(-TD_all_classes)), dim=1)


def compute_mean_adjacency_matrices2(net, comparison_data_labels_per_class,
                                     layer_names={'activation': 'act2', 'weight': 'dense3'}):
    '''
    :param model: pytorch sequential model.
    :param comparison_data_labels_per_class:
    :param layer_names: layers.

    :return: List of mean adjacency matrices per label.
    '''
    model_name = net.name
    ### added for hooks:
    weight_activation = {}
    if model_name == 'ResNet':
        model = net.model
        weight_activation['weight'] = getattr(model, layer_names['weight']).weight.detach()  # model.fc.weight.detach()
        mean_hook = getattr(model, layer_names['activation']).register_forward_hook(get_activation(weight_activation))
    else:
        weight_activation['weight'] = getattr(net, layer_names['weight']).weight.detach()
        mean_hook = getattr(net, layer_names['activation']).register_forward_hook(get_activation(weight_activation))

    all_labels = comparison_data_labels_per_class[1]
    comparison_data_classes = comparison_data_labels_per_class[0]

    mean_adjacency_matrices_per_label_list = []
    for cdata_cl in comparison_data_classes:
        net(cdata_cl)
        ## ResNet exception:
        if model_name == 'ResNet':
            weight_activation['activation'] = torch.flatten(weight_activation['activation'], 1)

        waa_mean = torch.mean(weight_activation['activation'], dim=0)
        mean_AM = activation_matrix_from_nodes_and_weights(waa_mean, weight_activation['weight'])
        MAM = torch.squeeze(mean_AM)
        mean_adjacency_matrices_per_label_list.append(MAM)

    mean_hook.remove()
    return mean_adjacency_matrices_per_label_list
