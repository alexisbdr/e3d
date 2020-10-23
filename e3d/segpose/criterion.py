import torch
from torch import nn
from utils.pose_utils import calc_vos_simple


class PoseCriterion(nn.Module):
    def __init__(
        self,
        t_loss_fn=nn.L1Loss(),
        q_loss_fn=nn.L1Loss(),
        sax=0.0,
        saq=0.0,
        srx=0,
        srq=0.0,
        learn_beta=False,
        learn_gamma=False,
    ):
        """
        Implements L_D from eq. 2 in the paper
        :param t_loss_fn: loss function to be used for translation
        :param q_loss_fn: loss function to be used for rotation
        :param sax: absolute translation loss weight
        :param saq: absolute rotation loss weight
        :param srx: relative translation loss weight
        :param srq: relative rotation loss weight
        :param learn_beta: learn sax and saq?
        :param learn_gamma: learn srx and srq?
        """
        super(PoseCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

    def forward(self, pred, targ):
        """
        :param pred: N x T x 6
        :param targ: N x T x 6
        :return:
        """

        # absolute pose loss
        s = pred.size()
        abs_loss = (
            torch.exp(-self.sax)
            * self.t_loss_fn(pred.view(-1, *s[2:])[:, :3], targ.view(-1, *s[2:])[:, :3])
            + self.sax
            + torch.exp(-self.saq)
            * self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:], targ.view(-1, *s[2:])[:, 3:])
            + self.saq
        )

        # get the VOs
        pred_vos = pose_utils.calc_vos_simple(pred)
        targ_vos = pose_utils.calc_vos_simple(targ)

        # VO loss
        s = pred_vos.size()
        vo_loss = (
            torch.exp(-self.srx)
            * self.t_loss_fn(
                pred_vos.view(-1, *s[2:])[:, :3], targ_vos.view(-1, *s[2:])[:, :3]
            )
            + self.srx
            + torch.exp(-self.srq)
            * self.q_loss_fn(
                pred_vos.view(-1, *s[2:])[:, 3:], targ_vos.view(-1, *s[2:])[:, 3:]
            )
            + self.srq
        )

        # total loss
        loss = abs_loss + vo_loss
        return loss
