import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PointnetSAModuleMSG(nn.Module):
    def __init__(self, npoint, radii, nsamples, mlps, use_xyz=True, bn=True):
        super().__init__()
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(nsamples)):
            self.groupers.append(pt_utils.QueryAndGroup(radius=radii[i], nsample=nsamples[i], use_xyz=use_xyz))

            layers = []
            last_channel = nsamples[i] * (3 if use_xyz else 0)
            for out_channel in mlps[i]:
                layers.append(pt_utils.Conv1d(last_channel, out_channel, bn=bn))
                layers.append(nn.BatchNorm1d(out_channel))
                layers.append(nn.ReLU())
                last_channel = out_channel
            self.mlps.append(nn.Sequential(*layers))

        self.attention_modules = nn.ModuleList([AttentionModule(mlps[i][-1]) for i in range(len(nsamples))])

    def forward(self, xyz, features=None):
        new_features_list = []
        for i in range(len(self.groupers)):
            new_xyz, new_features = self.groupers[i](xyz, features)
            new_features = self.mlps[i](new_features)
            new_features = self.attention_modules[i](new_features)  # Применение внимания
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super(AttentionModule, self).__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention_weights = self.softmax(attention_scores)
        output = torch.matmul(attention_weights, V)
        
        return output



class OA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_k = self.k_conv(x)                   # [B, da, N]        
        x_v = self.v_conv(x)                   # [B, de, N]

        energy = torch.bmm(x_q, x_k) / (math.sqrt(x_q.size(-1)))   # [B, N, N]
        attention = self.softmax(energy)                           # [B, N, N]

        x_r = torch.bmm(x_v, attention)  # [B, de, N]
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x


class SPCT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 128)

        self.oa_modules = nn.ModuleList([OA(128) for _ in range(4)])

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        oa_outputs = []
        for oa_module in self.oa_modules:
            x = oa_module(x)
            oa_outputs.append(x)
        
        x = torch.cat(oa_outputs, dim=1)
        x = self.linear(x)

        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(len(NPOINTS)):
            mlps = [ [channel_in] + mlp for mlp in MLPS[k] ]
            channel_out = sum([mlp[-1] for mlp in mlps])

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=NPOINTS[k],
                    radii=RADIUS[k],
                    nsamples=NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()
        for k in range(len(FP_MLPS)):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k])
            )

        cls_layers = [nn.Conv1d(FP_MLPS[0][-1], CLS_FC[0], 1, bias=False), nn.BatchNorm1d(CLS_FC[0]), nn.ReLU()]
        cls_layers.append(nn.Dropout(0.5))
        cls_layers.append(nn.Conv1d(CLS_FC[0], 1, 1))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        
        for sa_module in self.SA_modules:
            li_xyz, li_features = sa_module(l_xyz[-1], l_features[-1])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])

        pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()
        return pred_cls


def get_model(input_channels=0):
    return Pointnet2MSG(input_channels=input_channels)
