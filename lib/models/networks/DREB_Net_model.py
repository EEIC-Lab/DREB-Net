import torch.nn as nn
import numpy as np
import torch
import copy
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class RepVGGBlock_useSE(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, deploy=None,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
        super(RepVGGBlock_useSE, self).__init__()
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()
        self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)

        self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


class Deblur_Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Deblur_Up(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2 == None:
            return self.conv(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class MAGFF(nn.Module):
    def __init__(self, channels=128, r=4):
        super(MAGFF, self).__init__()
        internal_channels = int(channels // r)

        self.local_attention = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_attention_2 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        self.global_attention_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x_a = x1 + x2
        x_l = self.local_attention(x_a)
        x_g = self.global_attention(x_a)
        x_lg = x_l + x_g
        weight = self.sigmoid(x_lg)
        x_i = x1 * weight + x2 * (1 - weight)

        x_l2 = self.local_attention_2(x_i)
        x_g2 = self.global_attention_2(x_i)
        x_lg2 = x_l2 + x_g2
        weight2 = self.sigmoid(x_lg2)
        x_output = x1 * weight2 + x2 * (1 - weight2)
        return x_output


class LFAMM(nn.Module):
    def __init__(self, channels=128, height=128, weight=128):
        super(LFAMM, self).__init__()
        self.channels = channels
        self.height = height
        self.weight = weight
        self.learnable_h = self.height
        self.learnable_w = np.floor(self.weight/2).astype(int) + 1
        self.register_parameter('convolution' , torch.nn.Parameter(torch.rand(self.channels, self.learnable_h, self.learnable_w), requires_grad=True))

    def forward(self, x):
        x_fft = torch.fft.rfftn(x, dim=(-2, -1))
        x_fft = x_fft + 1e-8
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft)
        x_amp_invariant = torch.mul(x_amp, self.convolution)
        x_fft_invariant = x_amp_invariant * torch.exp(torch.tensor(1j) * x_pha)
        x_invariant = torch.fft.irfftn(x_fft_invariant, dim=(-2, -1) )
        return x_invariant


class DREB_Net(nn.Module):

    def __init__(self, num_blocks=[4, 6, 16, 1], width_multiplier=[1, 1, 1, 1], override_groups_map=None, deploy=False, use_checkpoint=False,
                 heads=None, head_conv=None):
        super(DREB_Net, self).__init__()
        self.deconv_with_bias = False
        self.heads = heads

        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_checkpoint = use_checkpoint

        self.LFAMM = LFAMM(channels=128, height=128, weight=128)

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock_useSE(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)

        # # used for deconv layers
        self.deconv_layers1 = self._make_deconv_layer(256, 4)
        self.deconv_layers2 = self._make_deconv_layer(256, 4)
        self.deconv_layers3 = self._make_deconv_layer(256, 4)

        self.deblur_down1 = Deblur_Down(64, 64)
        self.deblur_down2 = Deblur_Down(64, 128)
        self.deblur_down3 = Deblur_Down(128, 256)
        self.deblur_down4 = Deblur_Down(256, 512)
        self.deblur_up1 = Deblur_Up(512, 512, 256)
        self.deblur_up2 = Deblur_Up(256, 256, 128)
        self.deblur_up3 = Deblur_Up(128, 128, 64)
        self.deblur_up4 = Deblur_Up(64, 96, 64)
        self.deblur_up5 = Deblur_Up(64, 32, 3)

        self.MAGFF_attention = MAGFF(channels=128)

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))
            else:
                fc = nn.Conv2d(in_channels=256, out_channels=num_output, kernel_size=1, stride=1, padding=0)
            self.__setattr__(head, fc)


    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock_useSE(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                            stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)


    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding


    def _make_deconv_layer(self, num_filters, deconv_kernel):
        kernel, padding, output_padding = self._get_deconv_cfg(deconv_kernel)
        planes = num_filters
        layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.in_planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.in_planes = planes
        return layer


    def forward(self, x, mode):
        out = self.stage0(x)
        s0 = out
        s1 = self.deblur_down1(s0)
        s2 = self.deblur_down2(s1)

        for block in self.stage1:
            if self.use_checkpoint:
                out = checkpoint.checkpoint(block, out)
            else:
                out = block(out)

        for block in self.stage2:
            if self.use_checkpoint:
                out = checkpoint.checkpoint(block, out)
            else:
                out = block(out)

        out_LFAMM = self.LFAMM(out)

        out = self.MAGFF_attention(out, s2)

        out = out_LFAMM + out

        for block in self.stage3:
            if self.use_checkpoint:
                out = checkpoint.checkpoint(block, out)
            else:
                out = block(out)

        for block in self.stage4:
            if self.use_checkpoint:
                out = checkpoint.checkpoint(block, out)
            else:
                out = block(out)

        out = self.deconv_layers1(out)
        out = self.deconv_layers2(out)
        out = self.deconv_layers3(out)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(out)

        if mode == 'val':
            return [ret]
        
        elif mode == 'train':
            deblur_inp = s2
            down3 = self.deblur_down3(deblur_inp)
            down4 = self.deblur_down4(down3)
            up1 = self.deblur_up1(down4, down3)
            up2 = self.deblur_up2(up1, s2)
            up3 = self.deblur_up3(up2, s1)
            up4 = self.deblur_up4(up3, s0)
            deblur_out = self.deblur_up5(up4, None)

            return [ret], deblur_out
        else:
            raise ValueError("mode not eq train/val!!!")


def create_DREB_Net_detect(deploy=False, use_checkpoint=False, heads=None, head_conv=None):
    print('create_DREB_Net_detect')
    return DREB_Net(override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint, 
                  heads=heads, head_conv=head_conv)



if __name__ == '__main__':
    heads = {'hm': 9, 'wh': 2, 'reg': 2}
    head_conv = 64
    model = create_DREB_Net_detect(num_layers=None, heads=heads, head_conv=head_conv)
    input_size = 1024
    x = torch.randn(2, 3, input_size, input_size)
    print(x.shape)

    mode = 'val'
    if mode == 'train':
        y, deblur_out = model(x, mode)
        print(y[0]['hm'].shape)
        print(y[0]['wh'].shape)
        print(y[0]['reg'].shape)
        print(deblur_out.shape)
    elif mode == 'val':
        y = model(x, mode)
        print(y[0]['hm'].shape)
        print(y[0]['wh'].shape)
        print(y[0]['reg'].shape)


    # exit(0)
    total_params = sum(param.numel() for param in model.parameters())
    total_params_M = sum(param.numel() for param in model.parameters())/ 1e6
    print(f'模型参数数量: {total_params_M:.2f} MB')
    total_params_MB = total_params * 4 / (1024 * 1024)  # 参数量以MB为单位，每个参数占4字节（32位浮点数）
    print(f"模型参数内存占用量: {total_params_MB:.2f} MB")


    # 计算模型FLOPs
    from thop import profile
    input = torch.randn(1, 3, input_size, input_size)
    macs, params = profile(model.cpu(), inputs=(input,'val', ), verbose=False)
    total_flops_G = macs / (10**9)  # FLOPs以G为单位
    print(f"模型的总FLOPs: {total_flops_G:.2f} G")


    import time
    # 确保你的模型和输入都在GPU上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    your_model = model.to(device)
    # 生成一个随机的256x256x3张量
    input_tensor = torch.randn(1, 3, 1024, 1024).to(device)
    # 确保模型处于评估模式
    your_model.eval()
    # 预热GPU
    for _ in range(100):
        with torch.no_grad():
            _ = your_model(input_tensor, mode='val')

    # 设置计算次数和计时器
    num_iterations = 100
    total_time = 0

    # 进行多次前向传播
    for _ in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            output = your_model(input_tensor, mode='val')
        end_time = time.time()
        total_time += (end_time - start_time)

    # 计算并打印平均单帧推理时间
    average_inference_time = total_time / num_iterations 
    print(f'Average inference time on GPU: {average_inference_time*1000} ms')


    # # 在CPU上计算推理时间
    # device = torch.device('cpu')
    # your_model_cpu = model.to(device)
    # input_tensor_cpu = input_tensor.to(device)

    # # 预热CPU
    # for _ in range(100):
    #     with torch.no_grad():
    #         _ = your_model_cpu(input_tensor_cpu)

    # # 计算推理时间
    # total_time_cpu = 0
    # for _ in range(num_iterations):
    #     start_time = time.time()
    #     with torch.no_grad():
    #         output = your_model_cpu(input_tensor_cpu)
    #     end_time = time.time()
    #     total_time_cpu += (end_time - start_time)

    # average_inference_time_cpu = total_time_cpu / num_iterations
    # print(f'Average inference time on CPU: {average_inference_time_cpu*1000} ms')
    