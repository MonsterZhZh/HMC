import torch
from torch.utils.tensorboard import SummaryWriter

import time
import copy
from sklearn.metrics import confusion_matrix, average_precision_score


def net_train(model, hex, optimizer, scheduler, epochs, device, train_data_loader, test_data_loader, train_data_size, test_data_size, proportion):
    # Create Tensorboard
    writer = SummaryWriter('runs_hex_' + str(proportion))

    # Training loop
    best_acc = 0.0  # best accuracy on the test set
    best_epoch = 0  # corresponding epoch of the best accuracy
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        epoch_start = time.time()
        print('Epoch: {}/{}'.format(epoch, epochs) + '  Learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        model.train()

        train_loss = 0.0
        train_corrects = 0
        test_loss = 0.0
        test_corrects = 0
        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            fc_total = model(inputs)
            loss = hex(fc_total, labels, device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects = evaluation(train_corrects, hex, fc_total, labels, device)

        scheduler.step()

        # Evaluating on the test set in this epoch
        with torch.no_grad():
            model.eval()
            for j, (inputs, labels) in enumerate(test_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                fc_total = model(inputs)
                loss = hex(fc_total, labels, device)
                test_loss += loss.item() * inputs.size(0)
                test_corrects = evaluation(test_corrects, hex, fc_total, labels, device)

        avg_train_loss = train_loss / train_data_size
        # This needs to adjust according to the proportion ratio
        avg_train_acc = train_corrects.double() / (train_data_size * proportion)
        # avg_train_acc = 0.0
        avg_test_loss = test_loss / test_data_size
        avg_test_acc = test_corrects.double() / test_data_size

        writer.add_scalars('loss', {
            'training loss': avg_train_loss,
            'testing loss': avg_test_loss
        }, epoch)
        writer.add_scalars('Accuracy', {
            'Training Acc': avg_train_acc,
            'Testing Acc': avg_test_acc
        }, epoch)

        if best_acc < avg_test_acc:
            best_acc = avg_test_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

        epoch_end = time.time()
        print('Epoch: {:03d}, Training Loss: {:.4f}, Training Accuracy: {:.4f}%, Time: {:.4f}s'.format(epoch, avg_train_loss, avg_train_acc * 100, epoch_end - epoch_start))
        print('Epoch: {:03d}, Test Loss: {:.4f}, Test Accuracy: {:.4f}%'.format(epoch, avg_test_loss, avg_test_acc * 100))

    writer.close()
    print('Best Accuracy on the test set: {:.4f}% at epoch: {:03d}'.format(best_acc * 100, best_epoch))
    model.load_state_dict(best_model_wts)
    return model


def net_test(model, device, test_data_loader, num_classes, class_names, hex):
    # Overall Accuracy
    correct = 0
    total = 0
    # Accuracy of each class
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    # Confusion matrix
    y_true = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for j, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            leaf_labels = labels - 3  # CUB2011 class: 122-321; HRSC2016 class: 3-23
            fc_total = model(images)

            # batch_pMargin = hex.inference(fc_total, device)
            # batch_pMargin = batch_pMargin.data.T
            batch_pMargin = fc_total.data

            pMargin = batch_pMargin[:, 3:]    # CUB2011 class: 122-321; HRSC2016 class: 3-23
            _, predicted = torch.max(pMargin.data, 1)
            total += leaf_labels.size(0)
            correct += (predicted == leaf_labels).sum().item()
            c = (predicted == leaf_labels).squeeze()
            for i in range(leaf_labels.size(0)):
                label = leaf_labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                y_true.append(class_names[leaf_labels[i]])
                y_pred.append(class_names[predicted[i]])
    print('Overall Accuracy of the trained network on the test set: {:.2f}%'.format(100 * correct / total))
    for i in range(num_classes):
        print('Accuracy of {} : {:.2f}%'.format(class_names[i], 100 * class_correct[i] / class_total[i]))
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    print('Confusion Matrix:\n', cm)
    # plot_cm(cm, class_names, "Confusion Matrix")


def evaluation(corrects, hex, fc_total, labels, device):
    with torch.no_grad():
        leaf_labels = torch.nonzero(labels > 2)  # CUB2011 class: 122-321; HRSC2016 class: 3-23
        if leaf_labels.shape[0] > 0:
            select_leaf_labels = torch.index_select(labels, 0, leaf_labels.squeeze()) - 3
            select_fc_total = torch.index_select(fc_total, 0, leaf_labels.squeeze())
            # batch_pMargin = hex.inference(select_fc_total, device)
            # batch_pMargin = batch_pMargin.data.T
            batch_pMargin = select_fc_total.data
            pMargin = batch_pMargin[:, 3:]
            _, predictions = torch.max(pMargin.data, 1)
            corrects += torch.sum(predictions == select_leaf_labels.data)
        else:
            corrects += 0
    return corrects


def test_AP(model, device, test_set, test_data_loader, hex, mode):
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        model.eval()
        for j, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            select_labels = labels[:, test_set.to_eval]
            fc_total = model(images)

            if mode == 0:
                # Sig
                batch_pMargin = fc_total.data
            elif mode == 1:
                # Mar
                batch_pMargin = hex.inference(fc_total, device)
                batch_pMargin = batch_pMargin.data.T
            else:
                print('Unknown Test Mode!!!')

            predicted = batch_pMargin > 0.5
            total += select_labels.size(0) * select_labels.size(1)
            correct += (predicted.to(torch.float64) == select_labels).sum()
            cpu_batch_pMargin = batch_pMargin.to('cpu')
            y = select_labels.to('cpu')
            if j == 0:
                test = cpu_batch_pMargin
                test_y = y
            else:
                test = torch.cat((test, cpu_batch_pMargin), dim=0)
                test_y = torch.cat((test_y, y), dim=0)
        score = average_precision_score(test_y, test, average='micro')
        print('Accuracy:' + str(float(correct) / float(total)))
        print('Precision score:' + str(score))
