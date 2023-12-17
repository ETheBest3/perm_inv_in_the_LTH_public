## Cite PairNorm paper
import torch
from models import base
from pruning import sparse_global
import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc

import torch_geometric.utils.to_dense_adj as to_dense_adj


    
class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization 
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
           
            ('SCS'-mode is not in the paper but we found it works well in practice, 
              especially for GCN and GAT.)

            PairNorm is typically used after each graph convolution operation. 
        """
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on original data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
                
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


class Model(base.Model):
    def __init__(self, plan, initializer, outputs, criterion,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(Model, self).__init__()
        layers = []
        current_size = plan[0]
        for size in plan[1:]:
            layers.append(base.GraphConv(current_size, size))
            current_size=size
            
        self.hidden_layers = nn.ModuleList(layers)
        self.out_layer = base.GraphConv(plan[-1] , outputs)

        self.relu = nn.ReLU(True)
        self.norm = PairNorm(norm_mode, norm_scale)
        #for BCELoss
        if criterion == "BCELoss":
            self.criterion = nn.BCELoss()
            self.sigmoid = nn.Sigmoid()
        else:
            self.criterion = nn.CrossEntropyLoss()


    def forward(self, x, edge_index):
        adj = to_dense_adj(edge_index=edge_index, max_num_nodes=len(x)).squeeze(dim=0)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            
        x = self.out_layer(x, adj)
        #for BCELoss
        if isinstance(self.criterion, nn.BCELoss):
            x = self.sigmoid(x)
        return x
    
    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('gcn') and
                len(model_name.split('_')) > 1 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[1:]]))


    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=None, criterion="CrossEntropy"):
        """The name of a model is gcn_N1[_N2...].

        N1, N2, etc. are the number of neurons in each fully-connected layer excluding the
        output layer (number of classes neurons by default).
        
        N1 should be the number of features!!!
        """
        if criterion == "BCELoss":
            outputs = outputs or 1
        else:
            outputs = outputs or 2
            
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        plan = [int(n) for n in model_name.split('_')[1:]]
        return Model(plan, initializer, outputs, criterion)

    @property
    def output_layer_names(self):
        return ['out_layer.weight', 'out_layer.bias']
    
    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='gcn_1_10_100',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='graph_dataset',
            batch_size=64
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            lr=0.1,
            training_steps='40ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='out_layer.weight',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)


        
