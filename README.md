# GMM-EM
# Report for GMM-EM 

#### 金凯祺

`这是AI培训班的的第一次编程作业，主要任务为编写一个高斯混合模型，来对数据进行二分。`

`下面是任务说明：This is a binary classification task. Features used are 2-dimensional feature. You will use Gaussian Mixture Model to accomplish the task.`

#### GMM的初始化

`训练数据的格式为 Feature-Dim1 Feature-Dim2 Class-Label ，将数据根据label分为两类，对于每一类的数据训练高斯混合模型，由于之前观察过每一类的高斯混合模型可以看成四个单高斯的结合，因此将K设置为4，对于每一类的高斯混合模型进行训练。`

`由于GMM 对初始值敏感。故先用kmean方法聚类，利用聚类数据点的均值和斜方差作为各个高斯分量的初始值。`

#### 参数调整

`GMM主要是mu--均值多维数组，每行表示一个样本各个特征的均值`

​		     alpha--为模型响应度数组 

`​		     cov为协方差矩阵的数组  的调整`

`kmeans会传给GMM  mu ，alpha，cov的初值，随后使用EM算法训练GMM，改变mu，alpha，cov的值，如果mu的波动极小，则认为GMM以训练完毕。`

`对于dev的valid集的测试结果为0.97875，提交测试集的结果为0.98000`

#### 分析与讨论

`可以看到的是，kmeans对于GMM的提升是巨大的，如果用的是随机值，测试结果只能到0.6~0.7左右，不太理想。`

`GMM的求解办法基于EM算法，因此有可能陷于局部极值，这初始值的选取十分相关了。`

