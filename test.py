import torch
from torch.utils.data.dataloader import DataLoader
from data import PrepASV15Dataset, PrepASV19Dataset, PrepASV21Dataset
import torch.nn.functional as F
import eval_metrics as em
from tqdm import tqdm
from convnext_1d import convnext_1d_tiny
import numpy as np
import os
# import matplotlib.pyplot as plt


def asv_cal_accuracies(protocol, path_data, net, device, data_type='time_frame', dataset=19):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        
        softmax_acc = 0
        num_files = 0
        probs = torch.empty(0, 3).to(device)

        if dataset == 15:
            test_set = PrepASV15Dataset(protocol, path_data, data_type=data_type)
        elif dataset == 19:
            test_set = PrepASV19Dataset(protocol, path_data, data_type=data_type)
        else:
            test_set = PrepASV21Dataset(protocol, path_data, data_type=data_type)


        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)
        
        # f = open("models/score/A07-A19/A19.txt", "w")

        index = 1
        for test_batch in tqdm(test_loader):
            # load batch and infer

            test_sample, test_label, meta_label = test_batch

            num_files += len(test_label)

            # if num_files > 128:
            #     return 0

            test_sample = test_sample.to(device)
            test_label = test_label.to(device)
            infer = net(test_sample)

            # data = torch.split(infer, 1, dim=0)
            # for idx, sample in enumerate(data):
            #     var_path = os.path.join('T-SNE/Embeddings/MECA_Res2Net/bonafide', "%i.npy" % (index))
            #     index += 1
            #     np_var = sample.data.cpu().numpy() # 数据类型转换                
            #     np.save(var_path, np_var) 

            # obtain output probabilities
            t1 = F.softmax(infer, dim=1)
            t2 = test_label.unsqueeze(-1)
            row = torch.cat((t1, t2), dim=1)
            probs = torch.cat((probs, row), dim=0)

            # for i in range(len(meta_label)):
            #     f.write(str(meta_label[i])+' '+str(t1[i][0].item())+'\n')

            # calculate example level accuracy
            infer = infer.argmax(dim=1)
            batch_acc = infer.eq(test_label).sum().item()
            softmax_acc += batch_acc

        softmax_acc = softmax_acc / num_files
        # f.close()
    
    return softmax_acc, probs.to('cpu')


def cal_roc_eer(probs, show_plot=True):
    """
    probs: tensor, number of samples * 3, containing softmax probabilities
    row wise: [genuine prob, fake prob, label]
    TP: True Fake
    FP: False Fake
    """
    all_labels = probs[:, 2]
    zero_index = torch.nonzero((all_labels == 0)).squeeze(-1)
    # print(zero_index.shape)
    one_index = torch.nonzero(all_labels).squeeze(-1)
    # print(one_index.shape)
    zero_probs = probs[zero_index, 0]
    one_probs = probs[one_index, 0]
    threshold_index = torch.linspace(-0.1, 1.01, 10000)
    tpr = torch.zeros(len(threshold_index),)
    fpr = torch.zeros(len(threshold_index),)
    cnt = 0
    for i in threshold_index:
        tpr[cnt] = one_probs.le(i).sum().item()/len(one_probs)
        fpr[cnt] = zero_probs.le(i).sum().item()/len(zero_probs)
        cnt += 1

    sum_rate = tpr + fpr
    distance_to_one = torch.abs(sum_rate - 1)
    eer_index = distance_to_one.argmin(dim=0).item()
    out_eer = 0.5*(fpr[eer_index] + 1 - tpr[eer_index]).numpy()

    return out_eer


if __name__ == '__main__':

    test_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    protocol_file_path = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
    # protocol_file_path = 'spoof_15/CM_protocol/cm_evaluation.ndx'
    # protocol_file_path = 'LA/spoof.txt'
    # protocol_file_path = 'LA/ASVspoof2019_LA_cm_protocols/A07.txt'
    # protocol_file_path = 'T-SNE/bonafide.txt'
    data_path = 'LA/data/eval_6/'
    # data_path = 'spoof_15/data/eval_6/'

    # protocol_file_path = 'LA/trial_metadata.txt'
    # data_path = 'LA/data/2021_eval_6/'

    Net = convnext_1d_tiny(num_classes=2)
    num_total_learnable_params = sum(i.numel() for i in Net.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

    # check_point = torch.load('./trained_models/model/time_frame_37_ASVspoof2019_LA_Loss_0.0_dEER_0.16%_eEER_0.64%.pth', map_location='cuda:0')
    check_point = torch.load('./model/time_frame_37_ASVspoof2019_LA_Loss_0.0_dEER_0.16%_eEER_0.64%.pth', map_location='cuda:0')

    Net.load_state_dict(check_point['model_state_dict'])

    accuracy, probabilities = asv_cal_accuracies(protocol_file_path, data_path, Net, test_device, data_type='time_frame', dataset=19)
    print(accuracy * 100)

    eer = cal_roc_eer(probabilities)
    print(eer)

    print('End of Program.')

    
