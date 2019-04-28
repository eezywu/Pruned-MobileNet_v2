# Pruned-MobileNet_v2
Apply the pruning strategy of [Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) for MobileNet_v2.

## Results

The size of the input image is `224x224`.

#### Comparisons of different prune ratios
|  Step  | Prune Ratio | L1 value | Parameters | Top1 Accuracy (%) | Speed (FPS) |
| :---------------: | :------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
|    0     |  0   |  0   |       9.8MB        |        93.24        |         6.58         |
|    1     |  0  |  0.001   |     9.8MB        |        92.68        |         6.58         |
|    2     |  0.3  |  0.001   |            7.2MB            |        91.84        |         -         |
|    3     |  0.3 + 0.7 * 0.3 = 0.51  |  0.001   |            5.4MB            |        91.26        |         -         |
|    4     |  0.51 + 0.49 * 0.2 = 0.608 |  0  |            4.6MB            |        92.13        |         12.61         |
|    5     |  0.608 (merging BN)  |  -  |            4.5MB            |        92.13        |         17.24         |



#### Comparisons of speeds on different models

|  Model  | Speed on PC (FPS) | Speed on iPhone7p (FPS) |
| :---------------: | :------: | :--------------------------: |
|    ResNet50     |  1   |    3.6    |
|    MobileNet v2     |  6.58 |   24.14   |  
|    Pruned MobileNet v2 (with BN)     |  12.61  |  55.34    | 
|    Pruned MobileNet v2 (merging BN)     |  17.24  |  73.28<br>(200+ when the input is resized into `96x96` ) | 
