import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 读取文件并将每一行“?”转成数字5,去除“,”“\n”，再将每一行的处理后得数值放入list中并返回
def openFile(filename):
    datas = []
    with open(filename) as f:
        for line in f:
            data = []
            # 每一行“?”转成数字5,去除“,”“\n”
            data.extend(map(int, list(line.replace('?', '5').replace(',', '')\
                        .replace('\n', ''))))
            datas.append(data)
    datas = list(datas)
    
    return datas

# 将训练集的数据分成属性集trainData，和类别集trainLabel
def splitTrainSet(trainDataset):
	trainDataset = np.array(trainDataset)
	trainData = trainDataset[:, :-1].tolist()
	trainLabel = trainDataset[:, -1].tolist()
    
	return trainData, trainLabel

# 7194个样本，13个特征，9个类别
trainFilename = 'train.csv'
trainDataset = openFile(trainFilename)
# 1798个样本，13个特征
testFilename = 'test.csv'
testDataset = openFile(testFilename)

# 将训练集的数据集trainDataset分成属性集trainData，和类别集trainLabel
trainData, trainLabel = splitTrainSet(trainDataset)
# sklearn中的随机森林是基于RandomForestClassifier类实现的
# max_depth：决策树的最大深度为10
# min_samples_split：每次分裂节点是最小的分裂个数为10，即最小被分裂为10个
# min_samples_leaf：若某一次分裂时一个叶子节点上的样本数小于1，则会被剪枝
# n_estimators：随机森林中树的个数为50
# max_features：总的特征数为4
clf = RandomForestClassifier(max_depth=10, min_samples_split=10, \
                        min_samples_leaf=1, n_estimators=50, max_features=4)
# 先通过训练集训练出随机森林
clf.fit(trainData,  trainLabel)
# 通过训练好的随机森林对测试集进行预测
predictResult=clf.predict(testDataset)
# 将预测结果转成list
predictResult = list(predictResult)
# 得到预测结果的个数
m, = np.shape(predictResult)
# 将predictResult转成DataFrame类型并保存到testPredict.csv文件中
data = pd.DataFrame(predictResult, index=range(1, m+1))
data.to_csv('testPredict.csv')
