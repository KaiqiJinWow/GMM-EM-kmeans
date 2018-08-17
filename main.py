import matplotlib.pyplot as plt
from GMM import *
K =4 

Y = np.loadtxt('train.txt')
list1=[]
list2=[]
for i in range(Y.shape[0]):
    if 1.9<Y[i][2]: list2.append(Y[i]) 
    else: list1.append(Y[i])
matY1=np.matrix(list1,copy = True)
matY2=np.matrix(list2,copy = True)
matY1_nolabel=matY1[:,0:2]
matY2_nolabel=matY2[:,0:2]
matY = np.matrix(Y,copy = True)
matY_nolabel=matY[:,0:2]

mu1,cov1,alpha1 = GMM_EM(matY1_nolabel,K,50)
mu2,cov2,alpha2 = GMM_EM(matY2_nolabel,K,50)

C = np.loadtxt('dev.txt')
matC = np.matrix(C,copy=True)

matC_nolabel=matC[:,0:2]
N = C.shape[0]

dev_r1 = compute_r(alpha1, mu1, cov1, matC_nolabel)
dev_r2 = compute_r(alpha2, mu2, cov2, matC_nolabel)
right_num = 0
for i in range(N):
    if compute_s(alpha1, dev_r1[i], mu1, cov1, matC_nolabel[i]) > compute_s(alpha2, dev_r2[i], mu2, cov2, matC_nolabel[i]):
        if matC[i,2] == 1:
            right_num += 1
    else:
        if matC[i,2] == 2:
            right_num += 1

print( right_num/800)


tmp=np.loadtxt("test.csv", dtype=np.str, delimiter=",")
data = tmp[1:,1:].astype(np.float)
test_r1 = compute_r(alpha1, mu1, cov1,data)
test_r2 = compute_r(alpha2, mu2, cov2,data)
test_class = [["id",'classes']]
for i in range(data.shape[0]):
    if compute_s(alpha1, test_r1[i], mu1, cov1,data[i]) > compute_s(alpha2, test_r2[i], mu2, cov2, data[i]):
        test_class.append(([i,1]))
    else:
        test_class.append([i,2])

np.savetxt('new.csv', test_class, fmt='%s',delimiter = ',')
