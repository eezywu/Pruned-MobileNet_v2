# Prune-MobileNet_v2
Apply the pruning strategy of [Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) for MobileNet_v2.

## Results
### MobileNet v2
|  Step  | Prune Ratio | Parameters | Top1 Accuracy (%) | Speed (FPS) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
|    0     |  0(no l1)  |       9.8MB        |        93.24        |         93.78         |
|    1     |  0  |     9.8MB         |        92.68        |         93.78         |
|    2     |  0.3  |            7.2MB            |        91.84        |         2.25M         |
|    3     |  0.3 + 0.7 * 0.3 = 0.51  |            5.4MB            |        91.26        |         2.25M         |
|    4     |  0.51 + 0.49 * 0.2 = 0.608 |            4.6MB            |        92.13        |         2.25M         |
|    5     |  0.608(merge bn)  |            4.5MB            |        92.13        |         2.25M         |
