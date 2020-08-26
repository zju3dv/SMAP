"""
@author: Zhen Jianan
@contact: jnzhen99@163.com
"""
import torch
import torch.nn as nn
import torch.autograd as autograd


class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, output, rdepth):
        batch_size = output.size(0)
        output_num = output.size(1)
        assert output_num == 1
        loss, count = 0., 0
        for i in range(batch_size):
            for j in range(len(rdepth[i])):
                if rdepth[i, j, 2] > 0:
                    loss += torch.abs(output[i, 0, int(rdepth[i][j][0]), int(rdepth[i][j][1])] - rdepth[i][j][2])
                    count += 1
        if count == 0:   # used for forward
            loss += torch.abs(output[0, 0, 1, 1] - rdepth[0][0][2])
            loss = loss * 0
            count = 1
        return loss / count


class JointsL2Loss(nn.Module):
    def __init__(self, has_ohkm=False, topk=8, thres=0, paf_num=0):
        super(JointsL2Loss, self).__init__()
        self.has_ohkm = has_ohkm
        self.topk = topk
        self.paf_num = paf_num
        self.thres = thres
        self.calculate = nn.MSELoss(reduction='none')

    def forward(self, output, valid, label):
        assert output.shape == label.shape
        
        tmp_loss = self.calculate(output, label)
        tmp_loss = tmp_loss.mean(dim=[2, 3])
        weight = torch.gt(valid.squeeze(), self.thres).float()
        tmp_loss *= weight

        if not self.has_ohkm:
            loss = tmp_loss.mean()
        else:
            if self.paf_num == 0:
                topk_val, topk_id = torch.topk(tmp_loss, k=self.topk, dim=1, sorted=False)
                loss = topk_val.mean()
            else:
                keypoint_num = output.shape[1] - self.paf_num * 2
                keypoint_loss = tmp_loss[:, :keypoint_num]
                paf_loss = tmp_loss[:, keypoint_num:]
                keypoint_topk_val, keypoint_topk_id = torch.topk(keypoint_loss, k=self.topk, dim=1, sorted=False)
                paf_topk_val, paf_topk_id = torch.topk(paf_loss, k=self.topk*2, dim=1, sorted=False)
                loss = keypoint_topk_val.mean() + paf_topk_val.mean()
        
        return loss

       