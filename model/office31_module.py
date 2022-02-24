from torch import nn
from model.network import resnet
import torchvision


class FeatureExtractor(nn.Module):
    def __init__(self, net='resnet50', bottleneck_dim=1024):
        super(FeatureExtractor, self).__init__()
        self.resnet = resnet('resnet50')
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module('f_resnet50', self.resnet)

        self.bottleneck_layer_list = [nn.Linear(self.resnet.output_num(), bottleneck_dim),
                                      nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.bottleneck_layer(features)
        return features

class LabelClassifierFs(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=31):
        super(LabelClassifierFs, self).__init__()
        self.label_classifier = nn.Sequential()
        self.label_classifier.add_module('y_fc_1', nn.Linear(input_dim, hidden_dim))
        self.label_classifier.add_module('y_relu_1', nn.ReLU(True))
        self.label_classifier.add_module('y_drop_1', nn.Dropout(0.5))
        self.label_classifier.add_module('y_fc_2', nn.Linear(hidden_dim, output_dim))
        self.label_classifier.add_module('y_softmax', nn.LogSoftmax(dim=1))

        ## initialization
        self.label_classifier[0].weight.data.normal_(0, 0.01)
        self.label_classifier[0].bias.data.fill_(0.0)
        self.label_classifier[3].weight.data.normal_(0, 0.01)
        self.label_classifier[3].bias.data.fill_(0.0)


    def forward(self, input):
        label_out = self.label_classifier(input)
        return label_out.squeeze(1)


class LabelClassifierFt(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim1=1024, hidden_dim2=1024, output_dim=31):
        super(LabelClassifierFt, self).__init__()
        self.label_classifier = nn.Sequential()
        self.label_classifier.add_module('y_fc_1', nn.Linear(input_dim, hidden_dim1))
        self.label_classifier.add_module('y_bn_1', nn.BatchNorm1d(hidden_dim1))
        self.label_classifier.add_module('y_relu_1', nn.ReLU(True))
        self.label_classifier.add_module('y_drop_1', nn.Dropout(0.5))
        self.label_classifier.add_module('y_fc_2', nn.Linear(hidden_dim1, hidden_dim2))
        self.label_classifier.add_module('y_relu_2', nn.ReLU(True))
        self.label_classifier.add_module('y_drop_2', nn.Dropout(0.5))
        self.label_classifier.add_module('y_fc_3', nn.Linear(hidden_dim2, output_dim))
        self.label_classifier.add_module('y_softmax', nn.LogSoftmax(dim=1))

        ## initialization
        self.label_classifier[0].weight.data.normal_(0, 0.01)
        self.label_classifier[0].bias.data.fill_(0.0)
        self.label_classifier[4].weight.data.normal_(0, 0.01)
        self.label_classifier[4].bias.data.fill_(0.0)
        self.label_classifier[7].weight.data.normal_(0, 0.01)
        self.label_classifier[7].bias.data.fill_(0.0)

    def forward(self, input):
        label_out = self.label_classifier(input)
        return label_out.squeeze(1)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024):
        super(DomainDiscriminator, self).__init__()
        self.domain_discriminator = nn.Sequential()
        self.domain_discriminator.add_module('d_fc_1', nn.Linear(input_dim, hidden_dim))
        self.domain_discriminator.add_module('d_relu_1', nn.ReLU(True))
        self.domain_discriminator.add_module('d_dropout_1', nn.Dropout(0.5))
        self.domain_discriminator.add_module('d_fc_2', nn.Linear(hidden_dim, hidden_dim))
        self.domain_discriminator.add_module('d_relu_2', nn.ReLU(True))
        self.domain_discriminator.add_module('d_dropout_2', nn.Dropout(0.5))
        self.domain_discriminator.add_module('d_fc_3', nn.Linear(hidden_dim, 2))
        self.domain_discriminator.add_module('d_softmax', nn.LogSoftmax(dim=1))

        ## initialization
        self.domain_discriminator[0].weight.data.normal_(0, 0.01)
        self.domain_discriminator[0].bias.data.fill_(0.0)
        self.domain_discriminator[3].weight.data.normal_(0, 0.01)
        self.domain_discriminator[3].bias.data.fill_(0.0)
        self.domain_discriminator[6].weight.data.normal_(0, 0.03)
        self.domain_discriminator[6].bias.data.fill_(0.0)

    def forward(self, x):
        domain_out = self.domain_discriminator(x)
        return domain_out.squeeze()


class PredictionDiscriminator(nn.Module):
    def __init__(self, input_dim=31, hidden_dim=1024):
        super(PredictionDiscriminator, self).__init__()
        self.prediction_discriminator = nn.Sequential()
        self.prediction_discriminator.add_module('Df_fc_1', nn.Linear(input_dim, hidden_dim))
        self.prediction_discriminator.add_module('Df_relu1', nn.ReLU(True))
        self.prediction_discriminator.add_module('Df_Dropout1', nn.Dropout(0.5))
        self.prediction_discriminator.add_module('Df_fc_2', nn.Linear(hidden_dim, hidden_dim))
        self.prediction_discriminator.add_module('Df_relu2', nn.ReLU(True))
        self.prediction_discriminator.add_module('Df_Dropout2', nn.Dropout(0.5))
        self.prediction_discriminator.add_module('Df_fc_3', nn.Linear(hidden_dim, 2))
        self.prediction_discriminator.add_module('Df_softmax', nn.LogSoftmax(dim=1))

        ## initialization
        self.prediction_discriminator[0].weight.data.normal_(0, 0.01)
        self.prediction_discriminator[0].bias.data.fill_(0.0)
        self.prediction_discriminator[3].weight.data.normal_(0, 0.01)
        self.prediction_discriminator[3].bias.data.fill_(0.0)
        self.prediction_discriminator[6].weight.data.normal_(0, 0.03)
        self.prediction_discriminator[6].bias.data.fill_(0.0)

    def forward(self, x):
        prediction_output = self.prediction_discriminator(x)
        return prediction_output
