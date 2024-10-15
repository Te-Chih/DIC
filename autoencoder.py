from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from models import S1_Decoupled_XDMLP_GNN_COR_REC_CE, S1_Decoupled_ADMLP_GNN_COR_REC_BCE
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
import torch.nn as nn
import torch.nn.functional as F
import math
EPS = 1e-15
MAX_LOGSTD = 10

from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper.

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder.
    """
    def forward(
        self,
        z: Tensor,
        edge_index: Tensor = None,
        sigmoid: bool = True,
    ) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (torch.Tensor): The edge indices.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        if edge_index is None:
            return self.forward_all(z)
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj



class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (torch.nn.Module): The encoder module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder: Module, decoder: Optional[Module] = None,decoder2: Optional[Module] = None,criterion: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.decoder2 = InnerProductDecoder() if decoder2 is None else decoder2
        self.criterion = criterion
        GAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def decode2(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder2(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def x_recon_loss(self, z: Tensor, x_init: Tensor,mask:Tensor):
        x_rec = self.decoder2(z)
        loss = self.criterion(x_rec[mask], x_init[mask])

        return loss

    def x_test(self, z: Tensor, features_unmask_eval: Tensor,
             groundtruth_x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        # from sklearn.metrics import average_precision_score, roc_auc_score

        pre_x=self.decode2(z)
        eval_unmask_pre = pre_x[features_unmask_eval].clone()
        eval_unmask_pre = torch.where(eval_unmask_pre>0.5,1.0,0.0)
        eval_unmask_target =groundtruth_x[
            features_unmask_eval].clone()
        # 初始化指标
        accuracy_fuc = BinaryAccuracy().to(device)
        precision_fuc = BinaryPrecision().to(device)
        recall_fuc = BinaryRecall().to(device)
        f1score_fuc = BinaryF1Score().to(device)
        eval_unmask_accuracy_value = accuracy_fuc(eval_unmask_pre,
                                                  eval_unmask_target).to(device)
        eval_unmask_precision_value = precision_fuc(eval_unmask_pre,
                                                    eval_unmask_target).to(device)
        eval_unmask_recall_value = recall_fuc(eval_unmask_pre,
                                              eval_unmask_target).to(device)
        eval_unmask_f1score_value = f1score_fuc(eval_unmask_pre,
                                                eval_unmask_target).to(device)
        # return
        #
        # pos_y = z.new_ones(pos_edge_index.size(1))
        # neg_y = z.new_zeros(neg_edge_index.size(1))
        # y = torch.cat([pos_y, neg_y], dim=0)
        #
        # pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        # neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        # pred = torch.cat([pos_pred, neg_pred], dim=0)
        #
        # y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        #
        return eval_unmask_accuracy_value,eval_unmask_precision_value,eval_unmask_recall_value,eval_unmask_f1score_value

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
            and :math:`\log\sigma^2`.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        """"""  # noqa: D419
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


class ARGA(GAE):
    r"""The Adversarially Regularized Graph Auto-Encoder model from the
    `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.

    Args:
        encoder (torch.nn.Module): The encoder module.
        discriminator (torch.nn.Module): The discriminator module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        encoder: Module,
        discriminator: Module,
        decoder: Optional[Module] = None,
    ):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator
        reset(self.discriminator)

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.discriminator)

    def reg_loss(self, z: Tensor) -> Tensor:
        r"""Computes the regularization loss of the encoder.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(z))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss

    def discriminator_loss(self, z: Tensor) -> Tensor:
        r"""Computes the loss of the discriminator.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z.detach()))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss


class ARGVA(ARGA):
    r"""The Adversarially Regularized Variational Graph Auto-Encoder model from
    the `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.

    Args:
        encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
            and :math:`\log\sigma^2`.
        discriminator (torch.nn.Module): The discriminator module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        encoder: Module,
        discriminator: Module,
        decoder: Optional[Module] = None,
    ):
        super().__init__(encoder, discriminator, decoder)
        self.VGAE = VGAE(encoder, decoder)

    @property
    def __mu__(self) -> Tensor:
        return self.VGAE.__mu__

    @property
    def __logstd__(self) -> Tensor:
        return self.VGAE.__logstd__

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        return self.VGAE.reparametrize(mu, logstd)

    def encode(self, *args, **kwargs) -> Tensor:
        """"""  # noqa: D419
        return self.VGAE.encode(*args, **kwargs)

    def kl_loss(
        self,
        mu: Optional[Tensor] = None,
        logstd: Optional[Tensor] = None,
    ) -> Tensor:
        return self.VGAE.kl_loss(mu, logstd)

class DDGAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (torch.nn.Module): The encoder module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, num_nodes, in_size, hidden_size, out_size, num_layer,dropout, device,encoder: Module, decoder: Optional[Module] = None,decoder2: Optional[Module] = None,criterion: Optional[Module] = None):
        super().__init__()
        self.X_model_S1 = S1_Decoupled_XDMLP_GNN_COR_REC_CE(
            num_nodes=num_nodes,
            in_size=in_size,
            hidden_size=hidden_size,
            num_layer=num_layer,
            out_size=out_size,
            dropout=dropout,
            device=device)

        self.A_model_S1 = S1_Decoupled_ADMLP_GNN_COR_REC_BCE(
            num_nodes=num_nodes,
            in_size=in_size,
            hidden_size=hidden_size,
            num_layer=num_layer,
            out_size=out_size,
            dropout=dropout,
            device=device)
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.decoder2 = InnerProductDecoder() if decoder2 is None else decoder2
        self.criterion = criterion
        GAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self,feature, feature_mask, edge_index, edge_weight,num_neighbor=5, *args, **kwargs) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        new_feature, node_emb1, node_emb2 = self.X_model_S1(feature,
                                                            feature_mask,
                                                            edge_index,
                                                            edge_weight)
        new_A, A_emb1, A_emb2 = self.A_model_S1(feature, feature_mask,
                                                edge_index, edge_weight)
        # z = self.XA_model_S2(new_feature, new_data.train_pos_edge_index)
        return new_feature, node_emb1, node_emb2, new_A, A_emb1, A_emb2

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def decode2(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder2(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def x_recon_loss(self, z: Tensor, x_init: Tensor,mask:Tensor):
        x_rec = self.decoder2(z)
        loss = self.criterion(x_rec[mask], x_init[mask])

        return loss

    def x_test(self, z: Tensor, features_unmask_eval: Tensor,
             groundtruth_x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        # from sklearn.metrics import average_precision_score, roc_auc_score

        pre_x=self.decode2(z)
        eval_unmask_pre = pre_x[features_unmask_eval].clone()
        eval_unmask_pre = torch.where(eval_unmask_pre>0.5,1.0,0.0)
        eval_unmask_target =groundtruth_x[
            features_unmask_eval].clone()
        # 初始化指标
        accuracy_fuc = BinaryAccuracy().to(device)
        precision_fuc = BinaryPrecision().to(device)
        recall_fuc = BinaryRecall().to(device)
        f1score_fuc = BinaryF1Score().to(device)
        eval_unmask_accuracy_value = accuracy_fuc(eval_unmask_pre,
                                                  eval_unmask_target).to(device)
        eval_unmask_precision_value = precision_fuc(eval_unmask_pre,
                                                    eval_unmask_target).to(device)
        eval_unmask_recall_value = recall_fuc(eval_unmask_pre,
                                              eval_unmask_target).to(device)
        eval_unmask_f1score_value = f1score_fuc(eval_unmask_pre,
                                                eval_unmask_target).to(device)
        # return
        #
        # pos_y = z.new_ones(pos_edge_index.size(1))
        # neg_y = z.new_zeros(neg_edge_index.size(1))
        # y = torch.cat([pos_y, neg_y], dim=0)
        #
        # pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        # neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        # pred = torch.cat([pos_pred, neg_pred], dim=0)
        #
        # y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        #
        return eval_unmask_accuracy_value,eval_unmask_precision_value,eval_unmask_recall_value,eval_unmask_f1score_value

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


