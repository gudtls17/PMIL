import torch
import numpy as np
from os.path import join


def continus_mixup_data(*xs, y=None, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size()[0]
    index = torch.randperm(batch_size, device=torch.device('cuda'))
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y = lam * y + (1-lam) * y[index]
    return *new_xs, y


def mixup_data_by_class(x, nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    mix_xs, mix_nodes, mix_ys = [], [], []

    for t_y in y.unique():
        idx = y == t_y

        t_mixed_x, t_mixed_nodes, _, _, _ = continus_mixup_data(
            x[idx], nodes[idx], y[idx], alpha=alpha, device=device)
        mix_xs.append(t_mixed_x)
        mix_nodes.append(t_mixed_nodes)

        mix_ys.append(y[idx])

    return torch.cat(mix_xs, dim=0), torch.cat(mix_nodes, dim=0), torch.cat(mix_ys, dim=0)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_cluster_loss(matrixs, y, intra_weight=2):

    y_1 = y[:, 1]

    y_0 = y[:, 0]

    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = 0.0

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss


def inner_loss(label, matrixs):

    loss = 0

    if torch.sum(label == 0) > 1:
        loss += torch.mean(torch.var(matrixs[label == 0], dim=0))

    if torch.sum(label == 1) > 1:
        loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

    return loss


def intra_loss(label, matrixs):
    a, b = None, None

    if torch.sum(label == 0) > 0:
        a = torch.mean(matrixs[label == 0], dim=0)

    if torch.sum(label == 1) > 0:
        b = torch.mean(matrixs[label == 1], dim=0)
    if a is not None and b is not None:
        return 1 - torch.mean(torch.pow(a-b, 2))
    else:
        return 0

def connectivity_strength_thresholding(connectivity):
    """
    Left TopK connection in each region
    
    connectivity : (batch, ROI, ROI)
    """
    ConnMat_Thr_list = []

    for i, FC in enumerate(connectivity):
        # if i%50 ==0:
        #     print(i+1, '', end='', flush = True)
        index = np.abs(FC).argsort(axis=1)
        n_rois = FC.shape[0]
        dumy_FC = np.zeros((n_rois, n_rois))

        # Take only the top k correlates to reduce number of edges
        for j in range(n_rois):
            for k in range(n_rois - 10):
                dumy_FC[j, index[j, k]] = 0
            for k in range(n_rois - 10, n_rois):
                dumy_FC[j, index[j, k]] = FC[j, index[j,k]]
                
        ConnMat_Thr_list.append(dumy_FC)
    return np.array(ConnMat_Thr_list)

def get_ITS_wordreport(timescales, connectivity_thr, cc200_label_name):
    """
    Select brain region and change ITS value to word report
    
    timescales : (batch, ROI)
    connectivity_thr : (batch, ROI, ROI)
    cc200_label_name : (ROI)
    """
    
    its_top10_cc200_label_list = []
    its_top10_list = []

    for i in range(len(connectivity_thr)):
        # if i%50 ==0:
        #     print(i, '', end='', flush = True)
        its_top10_idx = np.where(np.abs(connectivity_thr[i]).sum(axis=1).argsort()<10)  # select topK degree centrality region index 
        its_top10_cc200_label = cc200_label_name[its_top10_idx]  # check label name of index
        its_top10 = timescales[i][its_top10_idx]  # check ITS value of index
        
        its_top10_cc200_label_list.append(its_top10_cc200_label)
        its_top10_list.append(its_top10)

    its_top10_cc200_label_list = np.array(its_top10_cc200_label_list)  # (batch, ROI)
    its_top10_list = np.array(its_top10_list)  # (batch, ROI)
    
    its_top10_degree = np.empty(its_top10_list.shape, dtype=object)  # degree template
    its_top10_degree[np.where(its_top10_list<=3)] = "short"
    its_top10_degree[np.where((its_top10_list>3) & (its_top10_list<=5))] = "intermediate"
    its_top10_degree[np.where(its_top10_list>5)] = "long"
    its_top10_degree[np.where(its_top10_degree==None)] = "long"
    
    its_word= its_top10_degree + " timescales " + its_top10_cc200_label_list.astype(object)
    its_word = np.sum(its_word[:, :-1] + ", ", axis=1) + its_word[:, -1]
    return its_word