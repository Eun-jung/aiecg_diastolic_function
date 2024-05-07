import numpy as np
from sklearn.metrics import roc_curve, auc

# Function for calculating the F1 score
def cal_F1(loader, model):
    correct = 0
    total = 0
    model.eval()
    cof_mat = np.zeros((4, 4))
    Ns = np.zeros(4)
    ns = np.zeros(4)
    for data, labels in loader:
        data_batch, label_batch = data.to(device), labels.to(device)
        outputs = F.softmax(model(data_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += label_batch.size(0)
        correct += predicted.eq(label_batch.view_as(predicted)).sum().item()
        acc = label_batch.view_as(predicted)
        for (a, p) in zip(acc, predicted):
            cof_mat[a][p] += 1
            Ns[a] += 1
            ns[p] += 1
    F1 = 0.0
    for i in range(len(Ns)):
        tempF = cof_mat[i][i] * 2.0 / (Ns[i] + ns[i])
        F1 = F1 + tempF
        print('F1' + str(i) + ':', tempF)
    F1 = F1 / 4.0
    print('cofmat', cof_mat)
    return 100 * correct / total, F1


def get_acc(prob, label):
    pred_label = prob.max(1, keepdim=True)[1]
    total = label.size(0)
    correct = pred_label.eq(label.view_as(pred_label)).sum().item()
    return correct / total


def get_auroc(prob, label):
    if prob.shape[1] > 2:
        auroc = get_auroc_for_multi_class(prob, label)
    elif prob.shape[1] == 2:
        auroc = get_auroc_for_binary(prob[:, 1], label[:, 1])
    else:
        assert False, "You should use at least 2 classes for your model"
    return auroc


def get_auroc_for_binary(prob, label):
    fpr, tpr, _ = roc_curve(label, prob)
    auroc = auc(fpr, tpr)
    return auroc


def get_auroc_for_multi_class(prob, label):
    fpr = dict()
    tpr = dict()
    auroc = dict()
    for i in range(prob.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], prob[:, i])
        auroc[i] = auc(fpr[i], tpr[i])
    return auroc

