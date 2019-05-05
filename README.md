# Pruned-MobileNet_v2

Apply the pruning strategy of [Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) for MobileNet_v2.

The Caffe implementation of the algorithm is available in [link](https://github.com/eezywu/Network-Slimming).

## Results

The size of the input image is `224x224`.

#### Comparisons of different prune ratios
|  Step  | Prune Ratio | L1 value | Parameters | Top1 Accuracy | Speed |
| :---------------: | :------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
|    0     |  0   |  0   |       9.8MB        |        93.24%        |         152.0ms         |
|    1     |  0  |  0.001   |     9.8MB        |        92.68%        |         152.0ms         |
|    2     |  0.3  |  0.001   |            7.2MB            |        91.84%        |         -         |
|    3     |  0.3 + 0.7 * 0.3 = 0.51  |  0.001   |            5.4MB            |        91.26%        |         -         |
|    4     |  0.51 + 0.49 * 0.2 = 0.608 |  0  |            4.6MB            |        92.13%        |         79.3ms         |
|    5     |  0.608 (merging BN)  |  -  |            4.5MB            |        92.13%        |         58.0ms         |



#### Comparisons of speeds on different models

|  Model  | Speed on PC (ms) | Speed on iPhone7p (ms) |
| :---------------: | :------: | :--------------------------: |
|    ResNet50     |  `$\approx$`1000ms   |    277.8ms    |
|    MobileNet v2     |  152.0ms |   41.4ms   |  
|    Pruned MobileNet v2 (with BN)     |  79.3ms  |  18.1ms    | 
|    Pruned MobileNet v2 (merging BN)     |  58.0ms  |  13.6ms<br>(`$\approx$`5ms when the input is resized into `96x96` ) | 

## Contact

If you have any problems, please feel free to contact me via eezywu@163.com.
