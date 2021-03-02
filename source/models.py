from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, dims_layers, dropout=False):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential()
        for i in range(1, len(dims_layers)):
            self.encoder.add_module(f'enc_dense{i:d}', nn.Linear(dims_layers[i - 1], dims_layers[i]))
            self.encoder.add_module(f'enc_relu{i:d}', nn.ReLU(True))
            if dropout:
                self.encoder.add_module(f'enc_dropout{i:d}', nn.Dropout(0.25))

        self.decoder = nn.Sequential()
        size = len(dims_layers) - 1
        for i in range(size - 1, 0, -1):
            self.decoder.add_module(f'dec_dense{size - i:d}', nn.Linear(dims_layers[i + 1], dims_layers[i]))
            self.decoder.add_module(f'dec_relu{size - i:d}', nn.ReLU(True))
            if dropout:
                self.decoder.add_module(f'dec_dropout{size - i:d}', nn.Dropout(0.25))
        self.decoder.add_module(f'dec_dense{size:d}', nn.Linear(dims_layers[1], dims_layers[0]))
        self.decoder.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        output_encoder = self.encoder(x)
        output_decoder = self.decoder(output_encoder)
        return output_encoder, output_decoder

    def forward_encoder(self, x):
        return self.encoder(x)
    
    def reset_parameters(self):
        for layer in self.encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class Classifier(nn.Module):
    def __init__(self, dims_layers, dropout=False):
        super(Classifier, self).__init__()
        self.output = nn.Sequential()
        for i in range(1, len(dims_layers)):
            self.output.add_module(f'dense{i:d}', nn.Linear(dims_layers[i - 1], dims_layers[i]))
            self.output.add_module(f'reslu{i:d}', nn.ReLU(True))
            if dropout:
                self.output.add_module(f'dec_dropout{i:d}', nn.Dropout(0.2))
        self.output.add_module(f'dense{len(dims_layers):d}', nn.Linear(dims_layers[-1], 1))
        self.output.add_module('sigmoid', nn.Sigmoid())

    def forward(self, z):
        return self.output(z)

    def reset_parameters(self):
        for layer in self.output.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
