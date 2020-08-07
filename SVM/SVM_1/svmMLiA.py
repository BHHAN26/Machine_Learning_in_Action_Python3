# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 22:26:59 2020

@author: B.H.HAN
"""


import numpy as np
import matplotlib.pyplot as plt
import random


'''
Function:
    Read Data：opens up the file and parses each line into class labels, 
and our data matrix.
    
Parameters:
    fileName
    
Reruens:
    dataMat
    labelMat
    
Modify:
    Jul 30 2020
'''
def loadDataSet(fileName):
    dataMat = []    # data matrix
    labelMat = []   # label matrix
    fr = open(fileName) # open file
    
    # Read line-by-line
    for line in fr.readlines():
        # slice each row, based on '\t'. Meaningtime, delete '\t'
        lineArr = line.strip().split('\t')  
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # append data
        labelMat.append(float(lineArr[2]))  #append label
    return dataMat, labelMat


'''
Function:
    select alpha_j randomly

Parameters:
    i - alpha
    m - number of alpha
    
Returns:
    j - return selected number 
    
Modify:
    Aug 03 2020
'''
def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))   #
    return j


'''
Function:
    clips alpha values that are greater than H or
less than L

Parameters:
    aj - value of alpha
    H - max of alpha
    L - min of alpha
    
Returns:
    aj
    
Modify:
    Aug 03 2020
'''
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


'''
Function:
    The simplified SMO algorithm
    
Parameters:
    dataMatIn - data matrix
    classLabels - data labels
    C - slack variable
    toler - fault tolerance rate
    maxIter - maximum iteration number
    
Returns:
    None
    
Modify:
    Aug 03 2020
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)   # np.[100,2]
    labelMat = np.mat(classLabels).transpose()
    b = 0; m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter_num = 0
    while(iter_num < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fxi - float(labelMat[i])
            if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fxj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if(L == H):
                    print('L == H')
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('eta>=0')
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print('alpha_j变化太小')
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[i, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if(0 < alphas[i] < C):
                    b = b1
                elif(0 < alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("第%d次迭代 样本：%d， alpha优化次数：%d" % (iter_num, i, alphaPairsChanged))
        if(alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数：%d" % iter_num)
    return b, alphas
   
 
"""
函数说明：计算w
Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签
    alphas - alphas值
    
Returns:
    w - 直线法向量
Modify:
    2018-07-24
"""
def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    # 我们不知道labelMat的shape属性是多少，
    # 但是想让labelMat变成只有一列，行数不知道多少，
    # 通过labelMat.reshape(1, -1)，Numpy自动计算出有100行，
    # 新的数组shape属性为(100, 1)
    # np.tile(labelMat.reshape(1, -1).T, (1, 2))将labelMat扩展为两列(将第1列复制得到第2列)
    # dot()函数是矩阵乘，而*则表示逐个元素相乘
    # w = sum(alpha_i * yi * xi)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


"""
函数说明：分类结果可视化
Returns:
    dataMat - 数据矩阵
    w - 直线法向量
    b - 直线截距
    
Returns:
    None
Modify:
    2018-07-23
"""
def showClassifer(dataMat, w, b):
    # 正样本
    data_plus = []
    # 负样本
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 转换为numpy矩阵
    data_plus_np = np.array(data_plus)
    # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)
    # 正样本散点图（scatter）
    # transpose转置
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    # 负样本散点图（scatter）
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    # enumerate在字典上是枚举、列举的意思
    for i, alpha in enumerate(alphas):
        # 支持向量机的点
        if(abs(alpha) > 0):
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolors='red')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, w, b)
    
        

        
        
        
        
        
        
        
        