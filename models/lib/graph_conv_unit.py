import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class _Collection_Unit(nn.Module):
    def __init__(self, dim, dim_lr, use_bn=False):
        super(_Collection_Unit, self).__init__()
        self.dim = dim
        self.dim_lr = dim_lr
        self.fc_lft = nn.Linear(dim, self.dim_lr, bias=True)
        self.fc_rgt = nn.Linear(self.dim_lr, dim, bias=True)
        normal_init(self.fc_lft, 0, 0.001)
        normal_init(self.fc_rgt, 0, 0.001) 
        self.relu = nn.ReLU(inplace=True)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.dim)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, target, source, attention_base):
        fc_left_out = self.fc_lft(source)
        fc_out = self.fc_rgt(fc_left_out)
        if self.use_bn:
            fc_out = self.bn(fc_out.view(-1, self.dim)).view(source.size(0), source.size(1), self.dim)

        collect = torch.bmm(attention_base, fc_out) 
        collect_avg = collect / (attention_base.sum(2).view(collect.size(0), collect.size(1), 1) + 1e-7) 
        return self.relu(collect_avg)

class _GraphConvolutionLayer_Collect(nn.Module):
    """ collect information from neighbors """
    def __init__(self, dim, dim_lr,use_bn=False):
        super(_GraphConvolutionLayer_Collect, self).__init__()
        self.collect_units = nn.ModuleList()
        self.collect_units.append(_Collection_Unit(dim, dim_lr, use_bn=use_bn)) # obj (subject) from rel
        self.collect_units.append(_Collection_Unit(dim, dim_lr, use_bn=use_bn)) # obj (object) from rel
        self.collect_units.append(_Collection_Unit(dim, dim_lr, use_bn=use_bn)) # rel from obj (subject)
        self.collect_units.append(_Collection_Unit(dim, dim_lr, use_bn=use_bn)) # rel from obj (object)

    def forward(self, target, source, attention, unit_id):
        collection = self.collect_units[unit_id](target, source, attention)
        return collection
