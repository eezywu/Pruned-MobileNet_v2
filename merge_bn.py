# coding:utf-8

import numpy as np
import sys
import os
import google.protobuf as pb
import google.protobuf.text_format
import sys
import caffe
import shutil
from copy import deepcopy

def load_and_fill_biases(src_model, src_weights, dst_model):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)
 
    for i, layer in enumerate(model.layer):
        if layer.type == 'Convolution' or layer.type == 'DepthwiseConvolution': # or layer.type == 'Scale':
            # 添加bias参数
            if layer.convolution_param.bias_term == False:
                layer.convolution_param.bias_term = True
                layer.convolution_param.bias_filler.type = 'constant'
                layer.convolution_param.bias_filler.value = 0.0
 
    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))
 
    caffe.set_mode_cpu()
    net_src = caffe.Net(src_model, src_weights, caffe.TEST)
    net_dst = caffe.Net(dst_model, caffe.TEST)
    for key in net_src.params.keys():
        for i in range(len(net_src.params[key])):
            net_dst.params[key][i].data[:] = net_src.params[key][i].data[:]
 
    return net_dst
 
 
def merge_conv_and_bn(net, i_conv, i_bn, i_scale):
    
    assert(i_conv != None)
    assert(i_bn != None)
 
    key_conv = net._layer_names[i_conv]
    key_bn = net._layer_names[i_bn]
    key_scale = net._layer_names[i_scale] if i_scale else None
 
    # Copy
    bn_mean = deepcopy(net.params[key_bn][0].data)
    bn_variance = deepcopy(net.params[key_bn][1].data)
    num_bn_samples = deepcopy(net.params[key_bn][2].data)
 
    # and Invalidate the BN layer
    net.params[key_bn][0].data[:] = 0
    net.params[key_bn][1].data[:] = 1
    net.params[key_bn][2].data[:] = 1
    if num_bn_samples[0] == 0:
        num_bn_samples[0] = 1
 
    if net.params.has_key(key_scale):
        print('Combine {:s} + {:s} + {:s}'.format(key_conv, key_bn, key_scale))
        scale_weight = deepcopy(net.params[key_scale][0].data)
        scale_bias = deepcopy(net.params[key_scale][1].data)
        net.params[key_scale][0].data[:] = 1
        net.params[key_scale][1].data[:] = 0
    else:
        print('Combine {:s} + {:s}'.format(key_conv, key_bn))
        scale_weight = 1
        scale_bias = 0
 
    weight = deepcopy(net.params[key_conv][0].data)
    bias = deepcopy(net.params[key_conv][1].data)
    alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + np.finfo(np.double).eps)
    net.params[key_conv][1].data[:] = bias * alpha + (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)
    for i in range(len(alpha)):
        net.params[key_conv][0].data[i] = weight[i] * alpha[i]
 
def merge_batchnorms_in_net(net):
    # for each BN
    for i, layer in enumerate(net.layers):
        if layer.type != 'BatchNorm':
            continue
 
        l_name = net._layer_names[i]
 
        l_bottom = net.bottom_names[l_name]
        assert(len(l_bottom) == 1)
        l_bottom = l_bottom[0]
        l_top = net.top_names[l_name]
        assert(len(l_top) == 1)
        l_top = l_top[0]
 
        can_be_absorbed = True
 
        # Search all (bottom) layers
        for j in xrange(i - 1, -1, -1):
            tops_of_j = net.top_names[net._layer_names[j]]
            if l_bottom in tops_of_j:
                if net.layers[j].type not in ['Convolution', 'DepthwiseConvolution']:
                    can_be_absorbed = False
                else:
                    # There must be only one layer
                    conv_ind = j
                    break
 
        if not can_be_absorbed:
            continue
 
        # find the following Scale
        scale_ind = None
        for j in xrange(i + 1, len(net.layers)):
            bottoms_of_j = net.bottom_names[net._layer_names[j]]
            if l_top in bottoms_of_j:
                if scale_ind:
                    # Followed by two or more layers
                    scale_ind = None
                    break
 
                if net.layers[j].type in ['Scale']:
                    scale_ind = j
 
                    top_of_j = net.top_names[net._layer_names[j]][0]
                    if top_of_j == bottoms_of_j[0]:
                        # On-the-fly => Can be merged
                        break
                else:
                    # Followed by a layer which is not 'Scale'
                    scale_ind = None
                    break
        merge_conv_and_bn(net, conv_ind, i, scale_ind)
    return net

def process_model(net, src_model, dst_model, func_loop, func_finally):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)

    for i, layer in enumerate(model.layer):
        map(lambda x: x(layer, net, model, i), func_loop)
 
    map(lambda x: x(net, model), func_finally)
 
    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))
 
# Functions to remove (redundant) BN and Scale layers
to_delete_empty = []
def pick_empty_layers(layer, net, model, i):
    if layer.type not in ['BatchNorm', 'Scale']:
        return
 
    bottom = layer.bottom[0]
    top = layer.top[0]
    if bottom != top and bottom + '/bn' != top:
        return

    if layer.type in ['BatchNorm', 'Scale']:
        print('Delete layer: {}'.format(layer.name))
        to_delete_empty.append(layer)
 
def remove_empty_layers(net, model):
    map(model.layer.remove, to_delete_empty)
 
# A function to add 'engine: CAFFE' param into 1x1 convolutions
def set_engine_caffe(layer, net, model, i):
    if layer.type == 'Convolution':
        if layer.convolution_param.kernel_size == 1\
            or (layer.convolution_param.kernel_h == layer.convolution_param.kernel_w == 1):
            layer.convolution_param.engine = dict(layer.convolution_param.Engine.items())['CAFFE']


if __name__ == '__main__':

    prototxt = "5_prune0.3_0.3_0.2_mobilenet_v2_deploy.prototxt"
    new_prototxt = '6_prune0.3_0.3_0.2_mobilenet_v2_bn_merged_deploy.prototxt'
    model = "5_prune0.3_0.3_0.2_mobilenet_v2_finetune.caffemodel"
    new_model = '6_prune0.3_0.3_0.2_mobilenet_v2_bn_merged.caffemodel'

    net = load_and_fill_biases(prototxt, model, prototxt + '.temp')
    net = merge_batchnorms_in_net(net)
 
    process_model(net, prototxt + '.temp', new_prototxt,
                  [pick_empty_layers, set_engine_caffe],
                  [remove_empty_layers])

    with open(new_prototxt) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)
    for i, layer in enumerate(model.layer):
        #### 处理非inplace操作的层 ####
        if layer.type != 'Input':
            for j in range(10):
                try:
                    model.layer[i].bottom[j] = model.layer[i].bottom[j].replace('/bn', '')
                except:
                    break
            for j in range(10):
                try:
                    model.layer[i].top[j] = model.layer[i].top[j].replace('/bn', '')
                except:
                    break

    with open(new_prototxt, 'w') as f:
        f.write(pb.text_format.MessageToString(model))

    # with open(new_prototxt, 'r') as f:
    #     lines = f.readlines()
    # with open(final_prototxt, 'w') as f:
    #     for line in lines:
    #         if '/bn' in line:
    #             line = line.replace('/bn', '')
    #         f.write(line)
    final_net = caffe.Net(new_prototxt, caffe.TEST)
    for name, layer in zip(net._layer_names, net.layers):
        if layer.type == 'DepthwiseConvolution' or layer.type == 'Convolution':
            new_weight = deepcopy(net.params[name][0].data[...])
            new_bias = deepcopy(net.params[name][1].data[...])
            new_name = name.replace('/bn', '')
            final_net.params[new_name][0].data[...] = deepcopy(new_weight)
            final_net.params[new_name][1].data[...] = deepcopy(new_bias)
            
    final_net.save(new_model)
