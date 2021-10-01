import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable

from .core import *
from .modules import *

class OCSoftmax(Model):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0, name = 'OCS', save_rate=0.9, loss_type = None):
        super(OCSoftmax, self).__init__(name = name)
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()
        self.loss_type = loss_type
        self.save_rate = save_rate

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        self.batch_size = x.size(0)
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        


        scores = x @ w.transpose(0,1)
        output_scores = scores

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        
        if self.loss_type == 'HardMining':
            loss = self.softplus(self.alpha * scores)
            ind_sorted = torch.argsort(-loss) # from big to small
            num_saved = int(self.save_rate * self.batch_size)
            ind_update = ind_sorted[:num_saved]
            loss_final = self.softplus(self.alpha*scores[ind_update]).mean()
            return loss_final, -output_scores.squeeze(1)

        else:

            loss = self.softplus(self.alpha * scores).mean()
            return loss, -output_scores.squeeze(1)

class AMSoftmax(Model):
    def __init__(self, num_classes, enc_dim, s=20, m=0.9, name = 'AMS'):
        super(AMSoftmax, self).__init__(name=name)
        self.enc_dim = enc_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, enc_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits

class MVSoftmax(Model):
    def __init__(self, fc_type='MV-AM', margin=0.35, t=0.2, scale=20, embedding_size=256, num_class=2,
                 easy_margin=False, name = 'MVS'):
        super(MVSoftmax, self).__init__(name = name)
        self.weight = Parameter(torch.Tensor(embedding_size, num_class))
        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.t = t
        self.easy_margin = easy_margin
        self.scale = scale
        self.fc_type = fc_type
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        assert self.fc_type in ['MV-Arc', 'MV-AM']#, 'RV-Arc', 'RV-AM']
        

    def forward(self, x, label):  # x (M, K), w(K, N), y = xw (M, N), note both x and w are already l2 normalized.
        kernel_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(x, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        batch_size = x.size(0)
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)  # ground truth score

        
        if self.fc_type == 'MV-AM':
            mask = cos_theta > gt - self.margin
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t  #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, gt - self.margin, gt)
            else:
                final_gt = gt - self.margin
        elif self.fc_type == 'MV-Arc':
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)

            mask = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m
                # final_gt = torch.where(gt > cos_theta_m, cos_theta_m, gt)
       
        else:
            raise Exception('unknown fc type!')

        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        cos_theta *= self.scale
        return cos_theta, final_gt

def loss_final(pred, label, loss_type, criteria, save_rate=0.9, gamma=2.0):
    if loss_type == 'Softmax':
        loss_final = criteria(pred, label)
    elif loss_type == 'FocalLoss':
        assert (gamma >= 0)
        input = F.cross_entropy(pred, label, reduce=False)
        pt = torch.exp(-input)
        loss = (1 - pt) ** gamma * input
        loss_final = loss.mean()
    elif loss_type == 'HardMining':
        batch_size = pred.shape[0]
        loss = F.cross_entropy(pred, label, reduce=False)
        ind_sorted = torch.argsort(-loss) # from big to small
        num_saved = int(save_rate * batch_size)
        ind_update = ind_sorted[:num_saved]
        loss_final = torch.sum(F.cross_entropy(pred[ind_update], label[ind_update]))
    else:
        raise Exception('unknown loss type!!')

    return loss_final
