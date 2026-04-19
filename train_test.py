
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter, \
    save_model_dict
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        data_tensor_list[i] = data_tensor_list[i].to(device)
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                        data_tensor_list[i][idx_dict["te"]].clone()), 0))
    labels = np.concatenate((labels_tr, labels_te))

    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"
    adj_train_list = []
    adj_test_list = []
    adj_tr_ind = []
    adj_te_ind = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_tr, adj_ind_tr = gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric)
        adj_train_list.append(adj_tr)
        adj_tr_ind.append(adj_ind_tr)

        adj_te, adj_ind_te = gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric)
        adj_test_list.append(adj_te)
        adj_te_ind.append(adj_ind_te)

    return adj_train_list, adj_test_list, adj_tr_ind, adj_te_ind


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i]))
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i + 1)].step()
        loss_dict["C{:}".format(i + 1)] = ci_loss.detach().cpu().numpy().item()
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(
                model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])))
        c = model_dict["C"](ci_list)
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()

    return loss_dict


def test_epoch(data_list, adj_list, te_idx, model_dict, one_hot_label):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]
    c = c[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()

    return prob


def train_test(data_folder, view_list, num_class, lr_e_pretrain, lr_e, lr_c, num_epoch_pretrain, num_epoch):
    test_inverval = 50

    l_relu = 0.4
    hidden_size = 32
    head = 3

    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)

 
    if data_folder == 'LGG2':
        adj_parameter = 8
        dim_he_list = [200, 200, 100]


    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    labels_trte_tensor = torch.LongTensor(labels_trte)
    onehot_labels_trte_tensor = one_hot_tensor(labels_trte_tensor, num_class)

    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)

    labels_tr_tensor = labels_tr_tensor.to(device)
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.to(device)

    onehot_labels_trte_tensor = onehot_labels_trte_tensor.to(device)
    sample_weight_tr = sample_weight_tr.to(device)

    adj_tr_list, adj_te_list, adj_tr_ind, adj_te_ind = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx,
                                                                        adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn, l_relu=l_relu,
                                 hidden_size=hidden_size, head=head)

    for tr in range(len(adj_tr_ind)):
        adj_tr_list[tr] = adj_tr_ind[tr].long().to(device)
    for te in range(len(adj_te_ind)):
        adj_te_list[te] = adj_te_ind[te].long().to(device)

    for m in model_dict:
        model_dict[m] = model_dict[m].to(device)
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)

    file_AUCs = 'result/' + data_folder + '/k/' + str(adj_parameter) + '/' + data_folder + "-" + time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.localtime()) + '.txt'
    file_best_AUCs = 'result/' + data_folder + '/k/' + str(adj_parameter) + '/' + data_folder + "-best" + '.csv'

    if num_class == 2:
        AUCs = ('Epoch\tACC\tF1\tAUC')
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')


    else:
        AUCs = ('Epoch\tACC\tF1 weighted\t F1 macro')
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')

    best_acc = 0.0
    best_f1 = 0.0
    best_auc = 0.0
    best_f1_weight = 0.0
    best_f1_macro = 0.0
    for epoch in range(num_epoch + 1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)

        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict, onehot_labels_trte_tensor)
            print("\nTest: Epoch {:d}".format(epoch))
            acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))

            if num_class == 2:
                f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                auc = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])
                print("Test ACC: {:.3f}".format(acc))
                print("Test F1: {:.3f}".format(f1))
                print("Test AUC: {:.3f}".format(auc))
                if acc > best_acc and f1 > best_f1 and auc > best_auc:
                    best_acc = acc
                    best_f1 = f1
                    best_auc = auc
                auc = [epoch, acc, f1, auc]
                save_auc(auc, file_AUCs)
            else:
                f1_weight = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
                f1_macro = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
                print("Test ACC: {:.3f}".format(acc))
                print("Test F1 weighted: {:.3f}".format(f1_weight))
                print("Test F1 macro: {:.3f}".format(f1_macro))
                if acc > best_acc and f1_weight > best_f1_weight and f1_macro > best_f1_macro:
                    best_acc = acc
                    best_f1_weight = f1_weight
                    best_f1_macro = f1_macro
                auc = [epoch, acc, f1_weight, f1_macro]
                save_auc(auc, file_AUCs)
            print()

    if num_class == 2:
        auc = ["best", best_acc, best_f1, best_auc]
        save_auc(auc, file_AUCs)
        save_best_auc(auc, file_best_AUCs)
    else:
        auc = ["best", best_acc, best_f1_weight, best_f1_macro]
        save_auc(auc, file_AUCs)
        save_best_auc(auc, file_best_AUCs)


def save_auc(auc, file_AUCs):
    with open(file_AUCs, 'a') as f:
        f.write('\t'.join(map(str, auc)) + '\n')


def save_best_auc(auc, file_best_AUCs):
    with open(file_best_AUCs, 'a+') as f:
        f.write('\t'.join(map(str, auc)) + '\n')
