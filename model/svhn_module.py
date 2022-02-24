from torch import nn
from model.gradientReverseLayer import GRL


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.dropout = nn.Dropout2d(p=0.6)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.feature_extractor = nn.Sequential()  # (b, 3, 32, 32)
        self.feature_extractor.add_module('f_instancenorm', nn.InstanceNorm2d(3))  # add for enhancing effect
        self.feature_extractor.add_module('f_conv_1', nn.Conv2d(3, 96, 3, padding=1))  # (b, 96, 32, 32)
        self.feature_extractor.add_module('f_bn_1', nn.BatchNorm2d(96))
        self.feature_extractor.add_module('f_lrelu_1', self.lrelu)
        self.feature_extractor.add_module('f_conv_2', nn.Conv2d(96, 96, 3, padding=1))
        self.feature_extractor.add_module('f_bn_2', nn.BatchNorm2d(96))
        self.feature_extractor.add_module('f_lrelu_2', self.lrelu)
        self.feature_extractor.add_module('f_conv_3', nn.Conv2d(96, 96, 3, padding=0))  # (b, 96, 30, 30)
        self.feature_extractor.add_module('f_bn_3', nn.BatchNorm2d(96))
        self.feature_extractor.add_module('f_lrelu_3', self.lrelu)
        self.feature_extractor.add_module('f_maxpool_1', nn.MaxPool2d(2, stride=(2, 2)))  # (b, 96, 15, 15)
        self.feature_extractor.add_module('f_dropout_1', nn.Dropout2d())

        self.feature_extractor.add_module('f_conv_4', nn.Conv2d(96, 192, 5, padding=2))  # (b, 192, 15, 15)
        self.feature_extractor.add_module('f_bn_4', nn.BatchNorm2d(192))
        self.feature_extractor.add_module('f_lrelu_4', self.lrelu)
        self.feature_extractor.add_module('f_conv_5', nn.Conv2d(192, 192, 5, padding=2))
        self.feature_extractor.add_module('f_bn_5', nn.BatchNorm2d(192))
        self.feature_extractor.add_module('f_lrelu_5', self.lrelu)
        self.feature_extractor.add_module('f_conv_6', nn.Conv2d(192, 192, 5, padding=0))  # (b, 192, 11, 11)
        self.feature_extractor.add_module('f_bn_6', nn.BatchNorm2d(192))
        self.feature_extractor.add_module('f_lrelu_6', self.lrelu)
        self.feature_extractor.add_module('f_maxpool_2', nn.MaxPool2d(2, stride=(2, 2)))  # (b, 192, 5, 5)
        self.feature_extractor.add_module('f_dropout_2', nn.Dropout2d())

    def forward(self, x):
        features = self.feature_extractor(x)
        return features.view(-1, 192 * 5 * 5)


class LabelClassifier(nn.Module):
    def __init__(self, input_dim=192 * 5 * 5, hidden_dim=500, output_dim=10):
        super(LabelClassifier, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.5)
        self.label_classifier = nn.Sequential()
        self.label_classifier.add_module('y_fc_1', nn.Linear(input_dim, hidden_dim))
        self.label_classifier.add_module('y_lrelu_1', self.lrelu)
        self.label_classifier.add_module('y_fc_2', nn.Linear(hidden_dim, output_dim))
        self.label_classifier.add_module('y_bn', nn.BatchNorm1d(output_dim))
        self.label_classifier.add_module('y_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        label_out = self.label_classifier(x)
        return label_out.squeeze()


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=192 * 5 * 5, hidden_dim1=500, hidden_dim2=100):
        super(DomainDiscriminator, self).__init__()
        self.domain_discriminator = nn.Sequential()
        self.domain_discriminator.add_module('d_fc_1', nn.Linear(input_dim, hidden_dim1))
        self.domain_discriminator.add_module('d_relu_1', nn.ReLU(True))
        self.domain_discriminator.add_module('d_fc_2', nn.Linear(hidden_dim1, hidden_dim2))
        self.domain_discriminator.add_module('d_relu_2', nn.ReLU(True))
        self.domain_discriminator.add_module('d_fc_3', nn.Linear(hidden_dim2, 2))
        self.domain_discriminator.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        domain_out = self.domain_discriminator(x)
        return domain_out.squeeze()


class PredictionDiscriminator(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=100):
        super(PredictionDiscriminator, self).__init__()
        self.prediction_discriminator = nn.Sequential()
        self.prediction_discriminator.add_module('Df_fc_1', nn.Linear(input_dim, hidden_dim))
        self.prediction_discriminator.add_module('Df_relu_1', nn.ReLU(True))
        self.prediction_discriminator.add_module('Df_fc_2', nn.Linear(hidden_dim, 2))
        self.prediction_discriminator.add_module('Df_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        prediction_output = self.prediction_discriminator(x)
        return prediction_output.squeeze()
