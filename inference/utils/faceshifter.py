import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import torch
import time
import cv2
from torch import nn
from torch.nn import (
    Linear, Conv2d, BatchNorm1d,
    BatchNorm2d, PReLU, ReLU,
    Sigmoid, Dropout, MaxPool2d,
    AdaptiveAvgPool2d, Sequential,
    Module, Parameter
)
import torch.nn.functional as F
from collections import namedtuple
import math
from .mtcnn import MTCNN

def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


def conv(c_in, c_out, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1, bias=False),
        norm(c_out),
        nn.LeakyReLU(0.1, inplace=True)
    )


class conv_transpose(nn.Module):
    def __init__(self, c_in, c_out, norm=nn.BatchNorm2d):
        super(conv_transpose, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = norm(c_out)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip):
        x = self.conv_t(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)


# Multilayer Attributes Encoder
class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        self.conv1 = conv(3, 32)
        self.conv2 = conv(32, 64)
        self.conv3 = conv(64, 128)
        self.conv4 = conv(128, 256)
        self.conv5 = conv(256, 256)

        self.conv_t1 = conv_transpose(256, 256)
        self.conv_t2 = conv_transpose(512, 128)
        self.conv_t3 = conv_transpose(256, 64)
        self.conv_t4 = conv_transpose(128, 32)

        self.apply(init_weights)

    def forward(self, Xt):
        enc1 = self.conv1(Xt)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)

        z_att1 = self.conv5(enc4)

        z_att2 = self.conv_t1(z_att1, enc4)
        z_att3 = self.conv_t2(z_att2, enc3)
        z_att4 = self.conv_t3(z_att3, enc2)
        z_att5 = self.conv_t4(z_att4, enc1)

        z_att6 = F.interpolate(z_att5, scale_factor=2, mode='bilinear', align_corners=True)

        return z_att1, z_att2, z_att3, z_att4, z_att5, z_att6


def conv_add(c_in, c_out):
    return nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False),
    )

class ADD(nn.Module):
    def __init__(self, c_x, c_att, c_id):
        super(ADD, self).__init__()

        self.c_x = c_x

        self.h_conv = nn.Conv2d(in_channels=c_x, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.att_conv1 = nn.Conv2d(in_channels=c_att, out_channels=c_x, kernel_size=1, stride=1, padding=0, bias=True)
        self.att_conv2 = nn.Conv2d(in_channels=c_att, out_channels=c_x, kernel_size=1, stride=1, padding=0, bias=True)

        self.id_fc1 = nn.Linear(c_id, c_x)
        self.id_fc2 = nn.Linear(c_id, c_x)

        self.norm = nn.InstanceNorm2d(c_x, affine=False)

    def forward(self, h, z_att, z_id):
        h_norm = self.norm(h)

        att_beta = self.att_conv1(z_att)
        att_gamma = self.att_conv2(z_att)

        id_beta = self.id_fc1(z_id)
        id_gamma = self.id_fc2(z_id)

        id_beta = id_beta.reshape(h_norm.shape[0], self.c_x, 1, 1).expand_as(h_norm)
        id_gamma = id_gamma.reshape(h_norm.shape[0], self.c_x, 1, 1).expand_as(h_norm)

        M = torch.sigmoid(self.h_conv(h_norm))
        A = att_gamma * h_norm + att_beta
        I = id_gamma * h_norm + id_beta

        return (torch.ones_like(M).to(M.device) - M) * A + M * I

class ADDResBlk(nn.Module):
    def __init__(self, c_in, c_out, c_att, c_id):
        super(ADDResBlk, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.add1 = ADD(c_in, c_att, c_id)
        self.conv1 = conv_add(c_in, c_in)
        self.add2 = ADD(c_in, c_att, c_id)
        self.conv2 = conv_add(c_in, c_out)

        if c_in != c_out:
            self.add3 = ADD(c_in, c_att, c_id)
            self.conv3 = conv_add(c_in, c_out)

    def forward(self, h, z_att, z_id):
        x = self.add1(h, z_att, z_id)
        x = self.conv1(x)
        x = self.add1(x, z_att, z_id)
        x = self.conv2(x)
        if self.c_in != self.c_out:
            h = self.add3(h, z_att, z_id)
            h = self.conv3(h)

        return x + h


class ADDGenerator(nn.Module):
    def __init__(self, c_id=512):
        super(ADDGenerator, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels=c_id, out_channels=256, kernel_size=2, stride=1, padding=0, bias=False)
        self.add1 = ADDResBlk(256, 256, 256, c_id)
        self.add2 = ADDResBlk(256, 256, 512, c_id)
        self.add3 = ADDResBlk(256, 256, 256, c_id)
        self.add4 = ADDResBlk(256, 128, 128, c_id)
        self.add5 = ADDResBlk(128, 64, 64, c_id)
        self.add6 = ADDResBlk(64, 3, 64, c_id)

        self.apply(init_weights)

    def forward(self, z_att, z_id):
        x = self.conv_t(z_id.reshape(z_id.shape[0], -1, 1, 1))
        x = self.add1(x, z_att[0], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add2(x, z_att[1], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add3(x, z_att[2], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add4(x, z_att[3], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add5(x, z_att[4], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add6(x, z_att[5], z_id)
        return torch.tanh(x)


class AEI_Net(nn.Module):
    def __init__(self, c_id=512):
        super(AEI_Net, self).__init__()
        self.encoder = MAE()
        self.generator = ADDGenerator(c_id=c_id)

    def forward(self, Xt, z_id):
        attr = self.encoder(Xt)
        Y = self.generator(attr, z_id)
        return Y, attr

    def get_attr(self, X):
        # with torch.no_grad():
        return self.encoder(X)

##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1 ,bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
                                       # )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self,x):
        feats = []
        x = self.input_layer(x)
        for m in self.body.children():
            x = m(x)
            feats.append(x)
        # x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x), feats
        # return x

##################################  MobileFaceNet #############################################################

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)

        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)
        return l2_norm(out)

##################################  Arcface head #############################################################

class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

##################################  Cosface head #############################################################

class Am_softmax(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self,embedding_size=512,classnum=51332):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = 0.35 # additive margin recommended by the paper
        self.s = 30. # see normface https://arxiv.org/abs/1704.06369
    def forward(self,embbedings,label):
        kernel_norm = l2_norm(self.kernel,axis=0)
        cos_theta = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1,1) #size=(B,1)
        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,label.data.view(-1,1),1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index] #only change the correct predicted output
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output


def swap_faces(Xs_raw, Xt_raw):
    detector = MTCNN()
    device = torch.device('cpu')
    G = AEI_Net(c_id=512)
    G.eval()
    G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=device))
    G = G.cpu()

    arcface = Backbone(50, 0.6, 'ir_se').to(device)
    arcface.eval()
    arcface.load_state_dict(torch.load('./saved_models/model_ir_se50.pth', map_location=device), strict=False)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    Xs_img = Image.fromarray(Xs_raw)
    Xs = detector.align(Xs_img, crop_size=(64, 64))

    Xs = test_transform(Xs)
    Xs = Xs.unsqueeze(0).cpu()
    with torch.no_grad():
        embeds, Xs_feats = arcface(F.interpolate(Xs, (112, 112), mode='bilinear', align_corners=True))
        embeds = embeds.mean(dim=0, keepdim=True)

    mask = np.zeros([64, 64], dtype=np.float)
    for i in range(64):
        for j in range(64):
            dist = np.sqrt((i-32)**2 + (j-32)**2)/32
            dist = np.minimum(dist, 1)
            mask[i, j] = 1-dist
    mask = cv2.dilate(mask, None, iterations=20)

    Xt_img = Image.fromarray(Xt_raw)

    Xt, trans_inv = detector.align(Xt_img, crop_size=(64, 64), return_trans_inv=True)

    Xt = test_transform(Xt)

    Xt = Xt.unsqueeze(0).cpu()
    with torch.no_grad():
        st = time.time()
        Yt, _ = G(Xt, embeds)
        Yt = Yt.squeeze().detach().cpu().numpy()
        st = time.time() - st
        print(f'inference time: {st} sec')
        Yt = Yt.transpose([1, 2, 0])*0.5 + 0.5
        Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderMode=cv2.BORDER_TRANSPARENT)
        mask_ = cv2.warpAffine(mask, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderMode=cv2.BORDER_TRANSPARENT)
        mask_ = np.expand_dims(mask_, 2)
        Yt_trans_inv = mask_*Yt_trans_inv + (1-mask_)*(Xt_raw.astype(np.float)/255.)

        merge = Yt_trans_inv

        return merge
