import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import build_mlp
from utils import gumbel_softmax, init_hidden
import numpy as np

class LiteEval(nn.Module):
    def __init__(self, args, num_classes=239):
        super(LiteEval, self).__init__()

        self.embedding_size = 2048
        self.small_embedding_size = 512
        self.num_layers = 1

        proj_img_kwargs = {
            'input_dim': 2048,
            'hidden_dims': (2048, ),
            'use_batchnorm': True,
            'dropout': 0,
        }

        small_proj_kwargs = {
            'input_dim': 1280,
            'hidden_dims': (512,),
            'use_batchnorm': True,
            'dropout': 0,
        }

        self.proj_mlp = build_mlp(**proj_img_kwargs)
        self.small_proj_mlp = build_mlp(**small_proj_kwargs)

        self.large_cell_size = args.large_cell_size
        self.small_cell_size = args.small_cell_size
        self.tau = args.tau
        self.num_classes = num_classes

        self.large_rnn = nn.LSTMCell(input_size=self.embedding_size + self.small_embedding_size,
                                     hidden_size=self.large_cell_size,
                                     bias=True)
        self.small_rnn = nn.LSTMCell(input_size=self.small_embedding_size,
                                     hidden_size=self.small_cell_size,
                                     bias=True)

        self.linear = nn.Linear(self.small_embedding_size + 2 * self.large_cell_size, 2)
        self.classifier = nn.Linear(self.large_cell_size, num_classes)
        self.num_steps = args.num_frames

    def forward(self, x, x_small, tau = 1.0):
        """
        :param x: [batch, len]
        :return: logits, r_stack
        """

        batch_size = x.size()[0]

        h_state_l = init_hidden(batch_size, self.large_cell_size)
        c_l = init_hidden(batch_size, self.large_cell_size)

        h_state_s = init_hidden(batch_size, self.small_cell_size)
        c_s = init_hidden(batch_size, self.small_cell_size)

        r_stack = []
        count = torch.zeros(batch_size)
        probs = self.classifier(h_state_l)

        tot_length = x.size(1) if self.num_steps >=128 else int(self.num_steps)

        for t in range(tot_length):
            features = x[:, t, :]
            features = features.contiguous().view(-1, self.embedding_size)
            small_features = x_small[:, t, :]
            small_features = small_features.contiguous().view(-1, 1280)

            features = self.proj_mlp(features)
            small_features = self.small_proj_mlp(small_features)
            h_state_s, c_s = self.small_rnn(small_features, (h_state_s, c_s))

            p_t = self.linear(torch.cat([small_features, h_state_l, c_l], 1))
            p_t = torch.log(F.softmax(p_t, dim=1))
            r_t = gumbel_softmax(self.training, p_t, tau, hard=True)
            
            h_state_l_new, c_l_new = self.large_rnn(torch.cat([features, small_features], dim=1), (h_state_l, c_l))


            partial = torch.cat([h_state_s[:, :self.small_cell_size],
                                 h_state_l[:, self.small_cell_size:self.large_cell_size]], dim=1)

            h_state_tilde = torch.transpose(torch.stack(
                            [h_state_l_new, partial], dim=2), 1, 2)

            r_stack.append(r_t)

            r_t = r_t.unsqueeze(2)
            r_t = r_t.expand(-1, -1, h_state_tilde.size()[2])

            c_tilde = torch.transpose(torch.stack(
                            [c_l_new,
                             torch.cat([c_s[:, :self.small_cell_size],
                                        c_l[:, self.small_cell_size:self.large_cell_size]],
                                       dim=1)
                             ], dim=2), 1, 2)

            h_state_l = r_t * h_state_tilde
            h_state_l = h_state_l.sum(dim=1)

            c_l = r_t*c_tilde
            c_l = c_l.sum(dim=1)


        r_stack = torch.stack(r_stack, dim=0).squeeze()
        logits = self.classifier(h_state_l)
        return logits, r_stack