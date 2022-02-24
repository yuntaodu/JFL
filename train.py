import copy
import math

import numpy as np
import torch

from model.models import DiscriminativeLoss

from test import testOurs


def trainOurModel(model, device, use_cuda, summarywriter, train_loader_source, train_loader_target, test_loader_target, 
            optimizer, scheduler, label_criterion, domain_criterion, prediction_criterion, history, dataset_name, 
            epoch, num_epoch, batch_size, num_classes, image_size, gamma, need_test=True):
    len_dataloader = min(len(train_loader_source), len(train_loader_target))
    # training
    model.train()
    label_criterion_s, label_criterion_t = copy.deepcopy(label_criterion), copy.deepcopy(label_criterion)
    epoch_label_loss_s, epoch_label_loss_t = 0., 0.
    epoch_domain_loss, epoch_prediction_loss, epoch_discriminative_loss = 0., 0., 0.

    for i, (batch_source, batch_target) in enumerate(zip(train_loader_source, train_loader_target)):
        n_iter = i + epoch * len_dataloader
        p = np.double(n_iter+1) / num_epoch/ len_dataloader  # define iteration ratio
        lamda = 2. / (1. + np.exp(-gamma * p)) - 1
        lambda_g = lamda
        lambda_f = lamda
        sntglamb = np.exp(-(1 - min( (n_iter+1-2000)/5000.0, 1)) * 10.0) if n_iter >= 2000 else 0.
        # custom fro svhn- mnist
        # sntglamb = 10 * np.exp(-(1 - min((n_iter + 1 - 2000) / 5000.0, 1)) * 10.0) if n_iter >= 2000 else 0.
        
        imgs_s, labels_s = batch_source
        imgs_s = torch.FloatTensor(batch_size, 3, image_size, image_size).copy_(imgs_s)
        imgs_t, labels_t = batch_target
        imgs_t = torch.FloatTensor(batch_size, 3, image_size, image_size).copy_(imgs_t)
        domain_labels_s = torch.zeros(batch_size).long()  # source domain: 0
        domain_labels_t = torch.ones(batch_size).long()  # target domain: 1
        domain_labels = torch.cat((domain_labels_s, domain_labels_t), dim=0)
        prediction_labels_s = torch.zeros(batch_size*2).long()  # source classifier: 0
        prediction_labels_t = torch.ones(batch_size*2).long()  # target classifier: 1
        prediction_labels = torch.cat((prediction_labels_s, prediction_labels_t), dim=0)

        if use_cuda:
            imgs_s, labels_s,  = imgs_s.to(device), labels_s.to(device)
            imgs_t, labels_t = imgs_t.to(device), labels_t.to(device)
            domain_labels = domain_labels.to(device)
            prediction_labels = prediction_labels.to(device)

        optimizer.zero_grad()  # pipe gradient
        # forward propagation
        label_out_s, label_out_t, domain_out, prediction_out = model(imgs_s, imgs_t, lambda_g, lambda_f)
        # compute label loss and domain loss
        label_loss_s = label_criterion_s(label_out_s, labels_s)
        # pseudo labeling for target domain
        pseudo_label_out = model.classifier_s(model.feature_extractor(imgs_t))
        pseudo_label_out = pseudo_label_out.detach()
        pseudo_label_t = pseudo_label_out.max(1, keepdim=False)[1].long()
        label_loss_t = label_criterion_t(label_out_t, pseudo_label_t)
        # compute domain adversarial loss and prediction adversarial loss
        domain_loss = domain_criterion(domain_out, domain_labels)
        prediction_loss = prediction_criterion(prediction_out, prediction_labels)

        Xs, Xt = model.feature_extractor(imgs_s), model.feature_extractor(imgs_t)
        dLoss = DiscriminativeLoss(Xs, Xt, labels_s, pseudo_label_t, num_classes, LAMBDA=30)

        epoch_label_loss_s += label_loss_s.data.item()
        epoch_label_loss_t += label_loss_t.data.item()
        epoch_domain_loss += domain_loss.data.item()
        epoch_prediction_loss += prediction_loss.data.item()
        epoch_discriminative_loss += dLoss.data.item()

        loss = 5 * label_loss_s + 0.5 * label_loss_t + domain_loss + prediction_loss + sntglamb * dLoss

        loss.backward()  # back propagation
        optimizer.step()  # parameters update

        if epoch == 0 and i == 0:
            summarywriter.add_graph(model, [imgs_s, imgs_t, torch.scalar_tensor(lambda_g), torch.scalar_tensor(lambda_f)])
        if i % 5 == 0:  # write to tensorboard
            with torch.no_grad():
                summarywriter.add_scalars('loss_each', {'label_loss_s': label_loss_s, 'label_loss_t': label_loss_t, 
                    'domain_loss': domain_loss, 'prediction_loss': prediction_loss,
                    'discriminative_loss': dLoss}, n_iter)
                closs = label_loss_s + label_loss_t + dLoss - (domain_loss + prediction_loss)
                summarywriter.add_scalar('loss', closs, n_iter)
                    
                # add test process ! necessary?
                pred_output_s = label_out_s.max(1, keepdim=True)[1]
                source_acc = labels_s.eq(pred_output_s.view_as(labels_s)).float().sum() / batch_size
                pred_output_t = label_out_t.max(1, keepdim=True)[1]
                target_acc = labels_t.eq(pred_output_t.view_as(labels_t)).float().sum() / batch_size
                summarywriter.add_scalars('train_acc', {'train_acc_s': source_acc, 'train_acc_t': target_acc}, n_iter)

    # print all losses and save model every epoch
    print('[epoch: %d / %d] loss: l_ys: %.4f, l_yt: %.4f, l_dg: %.4f, l_df: %.4f, l_d: %.4f' %
                (epoch, num_epoch, epoch_label_loss_s/len_dataloader, epoch_label_loss_t/len_dataloader, 
                epoch_domain_loss/len_dataloader, epoch_prediction_loss/len_dataloader, 
                epoch_discriminative_loss/len_dataloader))
    state = {'model': model.state_dict(), 'epoch': epoch}
    history['current_model'] = state  # save current model
    if need_test:
        # testing
        testOurs('revgrad', model, device, use_cuda, summarywriter, dataset_name, history, test_loader_target, epoch)
        model.train()
    summarywriter.close()
