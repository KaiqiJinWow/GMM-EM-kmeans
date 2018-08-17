import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
#!/usr/bin/python
# coding=utf-8
from numpy import *
def mul_norm(x, miu, cov):
 

    result = math.pow(linalg.det(cov), -0.5) / (2 * math.pi)
    temp = x-miu
    result *= exp(-0.5 * dot(dot(temp, linalg.inv(cov)), temp.T))
    return result

def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2-vector1, 2)))


# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i] = dataSet[index]
    return centroids
# 计算欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离

# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    clusterAssment = zeros((numSamples,2))
    clusterChanged = True

    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        for i in range(numSamples):
            minDist = 10000000.0
            minIndex = 0

            # find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j], dataSet[i])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # update its cluster
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i] = minIndex, minDist**2

        # update centroids
        for j in range(k):
            pointsInCluster = dataSet[clusterAssment[:,0] == j]
            centroids[j] = mean(pointsInCluster, axis = 0)

    # get the initialization for c and covs
    c = []
    covs = []
    for i in range(k):
        pointsInCluster = dataSet[clusterAssment[:,0] == i]
        c.append(len(pointsInCluster) * 1.0 / len(dataSet))
        covs.append(cov(pointsInCluster.T))
        
    return centroids, c, covs
def compute_s(c, r, miu, cov, x):
    """(list, list, array, array, array) -> float

    Return the probability of the data produced by a GMM.
    c, miu, cov are the parameters of the GMM
    """

    result = 0.0
    for k in range(len(c)):
        temp = x - miu[k]
        result += r[k] * math.log(c[k]) - 0.5 * r[k] * (math.log(linalg.det(cov[k])) +  dot(dot(temp, linalg.inv(cov[k])), temp.T))
    return result


def compute_L(c, r, miu, cov, tag_array):
    """(list, list, array, array, array) -> float

    Return the value of the likelihood function given parameters and data
    """

    result = 0.0
    for i in range(len(tag_array)):
        result += compute_s(c, r[i], miu, cov, tag_array[i])

    return result


def update_pram(c, r, miu, cov, tag_array):
    """(list, list, list(array), list(array), array) -> none

    Update the parameters of the GMM according the rules
    """

    r_sum = []
    for i in range(len(tag_array)):
        temp = 0.0
        for k in range(len(c)):
            temp += c[k] * mul_norm(tag_array[i], miu[k], cov[k])
        r_sum.append(temp)

    c_new = [0.0] * len(c)
    for i in range(len(tag_array)):
        for k in range(len(c)):
            r[i][k] = c[k] * mul_norm(tag_array[i], miu[k], cov[k]) / r_sum[i]
            c_new[k] += r[i][k]

    c = c_new

    for k in range(len(c)):
        tmp = array(r)[:,k]
        miu[k]= sum(array([tmp]).T * tag_array, axis=0) / c[k]
        cov[k] = dot(dot((tag_array - miu[k]).T, diag(tmp)), (tag_array-miu[k])) / c[k]

    c = [t/sum(c) for t in c]

def compute_r(c, miu, cov, tarray):
    """(list, list(array),list(array),array) -> list
    
    Return the r of dev data or test data
    r[n][m] is the poseterior probability of tarray[n] 
    belonging to component m
    """

    t_r = [[0.25 for col in range(len(c))] for row in range(len(tarray))]

    tr_sum = []
    for i in range(len(tarray)):
       temp = 0.0
       for k in range(len(c)):
          temp += c[k] * mul_norm(tarray[i], miu[k], cov[k])
       tr_sum.append(temp)

    for i in range(len(tarray)):
       for k in range(len(c)):
          t_r[i][k] = c[k] * mul_norm(tarray[i], miu[k], cov[k]) / tr_sum[i]
    return t_r
def kmeans_best(dataSet, k, n):
    best_miu, best_c, best_covs = kmeans(dataSet, k)
    best_r = compute_r(best_c, best_miu, best_covs, dataSet)
    best_L = compute_L(best_c, best_r, best_miu, best_covs, dataSet)    
    
    for i in range(1,n):
        temp_miu, temp_c, temp_covs = kmeans(dataSet, k)
        temp_r = compute_r(temp_c, temp_miu, temp_covs, dataSet)
        temp_L = compute_L(temp_c, temp_r, temp_miu, temp_covs, dataSet)   
        if temp_L > best_L:
            best_miu, best_c, best_covs = temp_miu, temp_c, temp_covs
        
    return best_miu, best_c, best_covs

def phi(Y, mu_k, cov_k):
    
    norm = multivariate_normal(mean=mu_k,cov=cov_k)
    return norm.pdf(Y)

def getExpectation(Y, mu, cov, alpha):   ##E-step
    N = Y.shape[0]
    K = 4

    gamma = np.mat(np.zeros((N,K)))

    prob = np.zeros((N,K))
    for k in range(K):
        prob[:,k] =phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    for k in range(K):
        gamma[:, k] = alpha[k]*prob[:,k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i,:])
    return gamma

def maxmize(Y, gamma):   ## M-step 
    N,D = Y.shape
    K = gamma.shape[1]

    mu = np.zeros((K,D))
    cov = []
    alpha = np.zeros(K)

    for k in range(K):
        Nk = np.sum(gamma[:,k])

        for d in range(D):
            mu[k,d] = np.sum(np.multiply(gamma[:,k],Y[:,d])) / Nk 
        
        cov_k = np.mat(np.zeros((D,D)))
        for i in range(N):
            cov_k += gamma[i,k] * (Y[i] - mu[k]).T * (Y[i] - mu[k]) /Nk
        cov.append(cov_k)
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu,cov,alpha

def scale_data(Y):
    for i in range(0,1):
        max_ = Y[:,i].max()
        min_ = Y[:,i].min()
        Y[:,i] = (Y[:,i] - min_) / (max_ - min_)
    return Y


def GMM_EM(Y,K,times):

    mu,alpha,cov = kmeans_best(Y,K,5)
    mu_s = 1e-6*(np.ones((mu.shape),dtype=float))
    mu_old = np.zeros(mu.shape)
    while(not(all(np.abs(mu_old-mu)<mu_s))):
        mu_old = mu
        print(abs(mu_old-mu))
        gamma = getExpectation(Y,mu,cov,alpha)
        mu,cov,alpha = maxmize(Y,gamma)
    return mu,cov,alpha


