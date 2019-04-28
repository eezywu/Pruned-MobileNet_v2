# coding:utf-8

import caffe
import numpy as np
import shutil
from copy import deepcopy

# model = "dw_mobilenet_v2/dw_mobilenet_v2_sl_0.001l1_part_right_iter_42500.caffemodel"
# prototxt = "dw_mobilenet_v2_deploy.prototxt"

model = "weights/2_prune0.3_mobilenet_v2.caffemodel"
prototxt = "models/2_prune0.3_mobilenet_v2_deploy.prototxt"

# model = "ResNet50/sl_ResNet50_sl_0.00005l1_right_iter_5000.caffemodel"
# prototxt = "ResNet_50_deploy.prototxt"

caffe.set_mode_gpu()
net = caffe.Net(prototxt, model, caffe.TEST)

percent = 0.3
new_prototxt = prototxt.replace('_mobilenet_v2', '{:.1f}_mobilenet_v2'.format(percent))
shutil.copyfile(prototxt, new_prototxt)
new_net = caffe.Net(new_prototxt, caffe.TEST)
total = 0
for name, layer in zip(net._layer_names, net.layers):
    # print(layer.type)
    if 'dwise' not in name and 'linear' not in name and layer.type == 'Scale':
        total += net.params[name][0].data.shape[0]
# print(total)
bn = np.zeros(total)
index = 0
for name, layer in zip(net._layer_names, net.layers):
    if 'dwise' not in name and 'linear' not in name and layer.type == 'Scale':
        size = net.params[name][0].data.shape[0]
        bn[index:(index+size)] = deepcopy(np.abs(net.params[name][0].data))
        index += size
# print(total)
# print(len(bn))

sorted_bn = np.sort(bn)
threshold_index = int(total * percent)
threshold = sorted_bn[threshold_index]

pruned = 0
cfg = []
cfg_mask = []
for k, (name, layer) in enumerate(zip(net._layer_names, net.layers)):
    if 'dwise' not in name and 'linear' not in name and layer.type == 'Scale':
        weight_copy = deepcopy(np.abs(net.params[name][0].data))
        one_index = np.squeeze(np.argwhere(weight_copy > threshold))
        mask = np.zeros_like(weight_copy)
        mask[one_index] = 1
        pruned = pruned + mask.shape[0] - np.sum(mask)
        net.params[name][0].data[...] *= mask
        net.params[name][1].data[...] *= mask
        cfg.append(int(np.sum(mask)))
        cfg_mask.append(deepcopy(mask))
        print('total channel: {:d} \t remaining channel: {:d} \t prune ratio: {:f}'.
            format(mask.shape[0], int(np.sum(mask)), np.sum(mask)/mask.shape[0]))
    elif layer.type == 'Scale':
        cfg.append(net.params[name][0].data.shape[0])
    elif layer.type == 'Pooling':
        cfg.append('M')

pruned_ratio = pruned/total

print('Pre-processing Successful!')

print(cfg)
print(threshold_index, threshold)

new_net = caffe.Net(prototxt, model, caffe.TEST)

layer_id_in_cfg = 0
start_mask = np.ones(3) # 一开始的卷积核是3个通道
end_mask = cfg_mask[layer_id_in_cfg]
for name, layer in zip(net._layer_names, net.layers):
    if 'dwise' not in name and 'linear' not in name and layer.type == 'Scale':
        # print(name)
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask))) # argwhere：找到非0索引；squeeze：去除维数为1的维度
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        
        #### prune bn ####
        bn_name = name.replace('scale', 'bn')
        new_weight = deepcopy(net.params[name][0].data[idx1.tolist()])
        new_bias = deepcopy(net.params[name][1].data[idx1.tolist()])
        new_running_mean = deepcopy(net.params[bn_name][0].data[idx1.tolist()])
        new_running_var = deepcopy(net.params[bn_name][1].data[idx1.tolist()])

        new_net.params[name][0].reshape(*new_weight.shape)
        new_net.params[name][1].reshape(*new_bias.shape)
        new_net.params[bn_name][0].reshape(*new_running_mean.shape)
        new_net.params[bn_name][1].reshape(*new_running_var.shape)

        new_net.params[name][0].data[...] = deepcopy(new_weight)
        new_net.params[name][1].data[...] = deepcopy(new_bias)
        new_net.params[bn_name][0].data[...] = deepcopy(new_running_mean)
        new_net.params[bn_name][1].data[...] = deepcopy(new_running_var)
        
        #### prune depthwise convolution ####
        dwise_conv_name = name.replace('expand/scale', 'dwise')
        if 'dwise' in dwise_conv_name and dwise_conv_name in net._layer_names:
            new_weight = deepcopy(net.params[dwise_conv_name][0].data[idx1.tolist(), :, :, :])
            new_net.params[dwise_conv_name][0].reshape(*new_weight.shape)
            new_net.params[dwise_conv_name][0].data[...] = new_weight

            output_size = new_weight.shape[0]
            prune_ratio = 1 - output_size*1.0/net.params[dwise_conv_name][0].data.shape[0]
            print('{:s}\tOut shape {:d}\t prune ratio:{:.2f}'.format(dwise_conv_name, output_size, prune_ratio))

            dwise_bn_name = name.replace('expand/scale', 'dwise/bn')
            dwise_scale_name = name.replace('expand/scale', 'dwise/scale')
            new_weight = deepcopy(net.params[dwise_scale_name][0].data[idx1.tolist()])
            new_bias = deepcopy(net.params[dwise_scale_name][1].data[idx1.tolist()])
            new_running_mean = deepcopy(net.params[dwise_bn_name][0].data[idx1.tolist()])
            new_running_var = deepcopy(net.params[dwise_bn_name][1].data[idx1.tolist()])

            new_net.params[dwise_scale_name][0].reshape(*new_weight.shape)
            new_net.params[dwise_scale_name][1].reshape(*new_bias.shape)
            new_net.params[dwise_bn_name][0].reshape(*new_running_mean.shape)
            new_net.params[dwise_bn_name][1].reshape(*new_running_var.shape)

            new_net.params[dwise_scale_name][0].data[...] = deepcopy(new_weight)
            new_net.params[dwise_scale_name][1].data[...] = deepcopy(new_bias)
            new_net.params[dwise_bn_name][0].data[...] = deepcopy(new_running_mean)
            new_net.params[dwise_bn_name][1].data[...] = deepcopy(new_running_var)

        #### prune output pointwise convolution ####
        linear_conv_name = name.replace('expand/scale', 'linear')
        if 'linear' in linear_conv_name and linear_conv_name in net._layer_names:
            # print(linear_name)
            # print(net.params[linear_name][0].data.shape)
            new_weight = deepcopy(net.params[linear_conv_name][0].data[:, idx1.tolist(), :, :])
            new_net.params[linear_conv_name][0].reshape(*new_weight.shape)
            new_net.params[linear_conv_name][0].data[...] = new_weight
            output_size = new_weight.shape[0]
            prune_ratio = 1 - output_size*1.0/net.params[linear_conv_name][0].data.shape[0]
            print('{:s}\tOut shape {:d}\t prune ratio:{:.2f}'.format(linear_conv_name, output_size, prune_ratio))

        layer_id_in_cfg += 1
        start_mask = deepcopy(end_mask)
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]

    #### prune input pointwise convolution or conv1 ####
    elif 'linear' not in name and layer.type == 'Convolution': # dwise为DepthwiseConvolution
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
        # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        #### mobilenet v2的卷积层没有bias ####

        if new_net.params[name][0].shape[0] == 143: # 如果是全连接层，不改变其输出个数
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            new_weight = deepcopy(net.params[name][0].data[:, idx0.tolist(), :, :])
        else: # 如果是卷积层
            if 'conv2_1/expand' in name: # 上一层是conv1，输出通道数可能会变
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                new_weight = deepcopy(net.params[name][0].data[:, idx0.tolist(), :, :])
                new_weight = deepcopy(new_weight[idx1.tolist(), :, :, :])
            else:
                new_weight = deepcopy(net.params[name][0].data[idx1.tolist(), :, :, :])
        new_net.params[name][0].reshape(*new_weight.shape)
        new_net.params[name][0].data[...] = new_weight

        output_size = new_weight.shape[0]
        prune_ratio = 1 - output_size*1.0/net.params[name][0].data.shape[0]
        print('{:s}\tOut shape {:d}\t prune ratio:{:.2f}'.format(name, output_size, prune_ratio))

layer_outputs = {}
for name, layer in zip(new_net._layer_names, new_net.layers):
    if 'Convolution' in layer.type:
        layer_outputs[name] = new_net.params[name][0].data.shape[0]
        # print(name, new_net.params[name][0].data.shape)
        # print(name, new_net.params[name][0].data.shape[0], net.params[name][0].data.shape[0])
# print(layer_outputs)

with open(new_prototxt, 'r') as f:
    lines = f.readlines()
new_prototxt = new_prototxt.replace('_mobilenet_v2', '{:.1f}_mobilenet_v2'.format(percent))
with open(new_prototxt, 'w') as f:
    for line in lines:
        if 'name:' in line:
            layer_name = line.split('"')[1]
        if 'num_output' in line:
            line = '    num_output: {:d}\n'.format(layer_outputs[layer_name])
        if 'group' in line:
            line = '    group: {:d}\n'.format(layer_outputs[layer_name])
        f.write(line)

train_prototxt = prototxt.replace('deploy', 'train')
with open(train_prototxt, 'r') as f:
    lines = f.readlines()
new_train_prototxt = new_prototxt.replace('deploy', 'train')
with open(new_train_prototxt, 'w') as f:
    for line in lines:
        if 'name:' in line:
            layer_name = line.split('"')[1]
        if 'num_output' in line:
            line = '    num_output: {:d}\n'.format(layer_outputs[layer_name])
        if 'group' in line:
            line = '    group: {:d}\n'.format(layer_outputs[layer_name])
        # if 'l1_lambda' in line:
        #     continue
        # if 'l1_lambda' in line and ('dwise' in layer_name or 'linear' in layer_name):
            # line = '    l1_lambda: 0.001\n'
        f.write(line)
new_net.save("prune{:.1f}_{:s}".format(percent, model.split('/')[-1]))
print(threshold_index, threshold)