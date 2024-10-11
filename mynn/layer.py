from torch import nn


def get_init_linear(in_feats, out_feats):
    linear = nn.Linear(in_feats, out_feats)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)
    return linear


def stack_layers(in_dim, arch):
    layers = nn.Sequential()
    for info in arch:
        out_dim = info['size']

        if info['layer'] == 'linear':
            layer = get_init_linear(in_dim, out_dim)
        else:
            layer = get_init_linear(in_dim, out_dim)
        layers.append(layer)

        if info['activation'] == 'relu':
            layers.append(nn.ReLU())
        elif info['activation'] == 'tanh':
            layers.append(nn.Tanh())

        in_dim = out_dim

    return layers
