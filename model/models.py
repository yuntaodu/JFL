from torch import nn
from model.gradientReverseLayer import GRL
import torch
import torch.nn.functional as F


class OurModel(nn.Module):
    def __init__(self, feature_extractor, label_classifier_s, label_classifier_t, domain_discriminator, prediction_discriminator):
        super(OurModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier_s = label_classifier_s
        self.classifier_t = label_classifier_t
        self.domain_discriminator = domain_discriminator
        self.prediction_discriminator = prediction_discriminator
    
    def forward(self,img_s, img_t, lambda_g, lambda_f):
        feature_s, feature_t = self.feature_extractor(img_s), self.feature_extractor(img_t)
        # Domain Discrimination
        features = torch.cat((feature_s, feature_t), dim=0)
        reverse_features = GRL.apply(features, lambda_g)  # reverse features
        domain_out = self.domain_discriminator(reverse_features)
        
        # Label Prediction for source and target domain
        label_out_s, label_out_t = self.classifier_s(feature_s), self.classifier_t(feature_t)

        # prediction by Classifier Ps and Classifier Pt
        prediction_s, prediction_t = self.classifier_s(features), self.classifier_t(features)
        predictions = torch.cat((prediction_s, prediction_t), dim=0)
        reverse_predictions = GRL.apply(predictions, lambda_f)  # reverse predictions
        prediction_out = self.prediction_discriminator(reverse_predictions)

        return label_out_s, label_out_t, domain_out, prediction_out


# add discriminative learning
def DiscriminativeLoss(Xs, Xt, Ys, pseudo_label_t, num_classes, LAMBDA=30):
    ys = F.one_hot(Ys, num_classes)  # (batch, num_class)
    yt = F.one_hot(pseudo_label_t, num_classes)
    graph_source = torch.sum(ys[:, None, :] * ys[None, :, :], 2)  # (batch, batch)
    distance_source = torch.mean((Xs[:, None, :] - Xs[None, :, :])**2, 2)  # (batch, batch)
    # LAMBDA = 30
    source_dLoss = torch.mean(graph_source * distance_source + (1 - graph_source) * F.relu(LAMBDA - distance_source))
    
    graph_target = torch.sum(yt[:, None, :] * yt[None, :, :], 2)
    distance_target = torch.mean((Xt[:, None, :] - Xt[None, :, :])**2, 2)
    target_dLoss = torch.mean(graph_target * distance_target + (1 - graph_target) * F.relu(LAMBDA - distance_target))

    current_source_count = torch.sum(ys, dim=0)
    current_target_count = torch.sum(yt, dim=0)
    current_positive_source_count = torch.clamp(current_source_count, min=1)
    current_positive_target_count = torch.clamp(current_target_count, min=1)

    current_source_centroid = torch.div(torch.sum(Xs[:, None, :] * ys[:, :, None], dim=0), current_positive_source_count[:, None])
    current_target_centroid = torch.div(torch.sum(Xt[:, None, :] * yt[:, :, None], dim=0), current_positive_target_count[:, None])

    # mask: balance class weight
    fm_mask = (current_source_count * current_target_count > 0).float()
    fm_mask /= torch.mean(fm_mask+1e-8)
    alignLoss = torch.mean(torch.mean(torch.square(current_source_centroid - current_target_centroid), 1)*fm_mask)
    discriminativeLoss = alignLoss + source_dLoss + target_dLoss
    return discriminativeLoss
