import torch
import torch.nn as nn
import point_utils


class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu'):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif activation == 'tanh':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                nn.BatchNorm2d(out_ch),
                nn.Tanh()
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()
            )


    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, activation='leakyrelu'):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU()
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.BatchNorm1d(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                self.ac
            )

    def forward(self, x):
        x = self.fc(x)
        return x



class transform_net(nn.Module):
    def __init__(self, in_ch, K=3):
        super(transform_net, self).__init__()
        self.K = K
        self.conv2d1 = conv_2d(in_ch, 64, 1)
        self.conv2d2 = conv_2d(64, 128, 1)
        self.conv2d3 = conv_2d(128, 1024, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(512, 1))
        self.fc1 = fc_layer(1024, 512)
        self.fc2 = fc_layer(512, 256)
        self.fc3 = nn.Linear(256, K*K)
    


    def forward(self, x):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1,self.K * self. K).repeat(x.size(0),1)
        device = x.device
        iden = iden.to(device) 
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x



class adapt_layer_off(nn.Module):
    def __init__(self, num_node=64, offset_dim=3, trans_dim_in=64, trans_dim_out=64, fc_dim=64):
        super(adapt_layer_off, self).__init__()
        self.num_node = num_node
        self.offset_dim = offset_dim
        self.trans = conv_2d(trans_dim_in, trans_dim_out, 1)
        self.pred_offset = nn.Sequential(
            nn.Conv2d(trans_dim_out, offset_dim, kernel_size=1, bias=False),
            nn.Tanh())
        self.residual = conv_2d(trans_dim_in, fc_dim, 1)

    def forward(self, input_fea, input_loc):
        # Initialize node
        fpoint_idx = point_utils.farthest_point_sample(input_loc, self.num_node)  # (B, num_node)
        fpoint_loc = point_utils.index_points(input_loc, fpoint_idx)              # (B, 3, num_node)
        fpoint_fea = point_utils.index_points(input_fea, fpoint_idx)              # (B, C, num_node)

        group_idx = point_utils.query_ball_point(0.3, 64, input_loc, fpoint_loc)  # (B, num_node, 64)
        group_fea = point_utils.index_points(input_fea, group_idx)                # (B, C, num_node, 64)
        group_fea = self.trans(group_fea)                                         # ✅ reduce C to trans_dim_out (e.g., 64)

        seman_trans = self.pred_offset(group_fea)  # (B, 3, num_node, 64)
        seman_trans = seman_trans.permute(0, 2, 3, 1)  # (B, num_node, 64, 3)

        fpoint_loc_trans = fpoint_loc.permute(0, 2, 1).unsqueeze(2) + seman_trans  # (B, num_node, 1, 3) + (B, num_node, 64, 3)
        fpoint_loc_trans = fpoint_loc_trans.permute(0, 3, 1, 2)  # (B, 3, num_node, 64)
        node_fea = self.residual(fpoint_fea.unsqueeze(-1))  # (B, fc_dim, num_node, 1)

        return fpoint_fea, node_fea, fpoint_loc_trans  # (raw features, transformed features, offset locations)

