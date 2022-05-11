import numpy as np
import matplotlib as plt
from matplotlib import pyplot
from scipy.stats import multivariate_normal
#1
def data_generation(num):
    def gaus_point(i):
        if(i==1):
            return np.random.multivariate_normal(M1,Cov1)
        return np.random.multivariate_normal(M2,Cov2)
    M1=np.array([-1,-1])
    M2=np.array([1,1])
    Cov1=np.array([(0.8,0),(0,0.8)])
    Cov2=np.array([(0.75,-0.2),(-0.2,0.6)])
    P_z=[0.7, 0.3]
    rand=np.random.choice([1,2],num,p=P_z)
    data=np.asanyarray([gaus_point(rand[i]) for i in range(len(rand))])
    group1=np.array([])
    group2 = np.array([])
    for i in range(len(data)):
        if(rand[i]==1):
            group1=np.append(group1,data[i])
        else:
            group2=np.append(group2,data[i])
    group1=group1.reshape(len(group1)//2,2)
    group2=group2.reshape(len(group2) // 2, 2)
    plt.pyplot.scatter(group1[:, 0], group1[:, 1], c="red", label="Group1")
    plt.pyplot.scatter(group2[:, 0], group2[:, 1], c="blue", label="Group2")
    plt.pyplot.title("GMM of 1000 points, before EM")
    plt.pyplot.legend()
    plt.pyplot.show()
    return data

#resulte=data_generation(1000)
#plt.pyplot.scatter(resulte[:,0],resulte[:,1])
#plt.pyplot.show()

#2
def Kmeans(epsilon,data_point=data_generation(50)):
    k=0
    #data_point=data_generation(50)
    center=np.asanyarray([np.random.uniform(-2,2) for i in range(4)]).reshape(2,2)
    plt.pyplot.scatter(data_point[:, 0], data_point[:, 1])
    plt.pyplot.scatter(center[0, 0], center[0, 1],c="purple")
    plt.pyplot.scatter(center[1, 0], center[1, 1],c="purple")
    plt.pyplot.annotate("Mean1",(center[0, 0], center[0, 1]))
    plt.pyplot.annotate("Mean2", (center[1, 0], center[1, 1]))
    plt.pyplot.title("START OF K-MEANS, RANDOM CENTERS\n center1=["+str(format(center[0][0],".3f"))+" "+str(
        format(center[0][1],".3f"))+"] center2=["+str(format(center[1][0],".3f"))+" "+str(format(center[1][1],".3f"))+"]")
    plt.pyplot.show()
    delta1=1
    delta2=1
    while((delta1>epsilon)|(delta2>epsilon)):
        mean1 = np.array([])
        mean2 = np.array([])
        for point in data_point:
            if(np.linalg.norm(point-center[0])<=np.linalg.norm(point-center[1])):
                mean1=np.append(mean1,(point[0],point[1]),axis=0)
            else:
                mean2=np.append(mean2,(point[0],point[1]),axis=0)
        mean1 = np.reshape(mean1, (len(mean1) // 2, 2))
        mean2 = np.reshape(mean2, (len(mean2) // 2, 2))
        new_min1=[float(sum(l))/len(l) for l in zip(*mean1)]
        new_min2=[float(sum(l))/len(l) for l in zip(*mean2)]
        print("iteration: ", k)
        k+=1
        print("old: ",center)
        delta1=np.linalg.norm(new_min1-center[0])
        delta2=np.linalg.norm(new_min2-center[1])
        center[0]=new_min1
        center[1]=new_min2
        print("delta 1:",delta1,"delta 2:",delta2)
        print("new:",center)
        print("---------------------------------------")
        plt.pyplot.scatter(center[0, 0], center[0, 1], c="purple")
        plt.pyplot.scatter(center[1, 0], center[1, 1], c="purple")
        plt.pyplot.annotate("Mean1", (center[0, 0], center[0, 1]))
        plt.pyplot.annotate("Mean2", (center[1, 0], center[1, 1]))
        plt.pyplot.scatter(mean1[:,0],mean1[:,1],c="red",label="Group1")
        plt.pyplot.scatter(mean2[:,0],mean2[:,1],c="blue",label="Group2")
        plt.pyplot.title("K-MEANS, ITERATION="+str(k)+"\ncenter1=[" + str(format(center[0][0], ".3f")) + " " + str(
            format(center[0][1], ".3f")) + "] center2=[" + str(format(center[1][0], ".3f")) + " " + str(
            format(center[1][1], ".3f")) + "]")
        plt.pyplot.legend()
        plt.pyplot.show()
    return center[0],center[1]

#Kmeans(0.0001,50)

#3
def EM(epsilon):
    iteration=0
    data_point=data_generation(1000)
    #init random gausian or KMEANS
    center1=np.asarray([np.random.uniform(-2,2) for i in range(2)])
    center2=np.asarray([np.random.uniform(-2,2) for i in range(2)])
    #center1,center2=Kmeans(epsilon,data_point)
    center=np.array([center1,center2])
    cov1=np.diag([np.random.uniform(0,1) for i in range(2)])
    cov2=np.diag([np.random.uniform(0,1) for i in range(2)])
    cov=np.array([cov1,cov2])
    P_z1=np.random.uniform(0,1)
    P_z2=1-P_z1
    P_z=np.array([P_z1,P_z2])
    gaus1=multivariate_normal(center1,cov1)
    gaus2=multivariate_normal(center2,cov2)
    gaus = [gaus1, gaus2]
    print("----------iteration num ", iteration, "---------")
    print("1: \n",cov1," \n",center1," \n",P_z1)
    print("2: \n", cov2, " \n", center2, " \n", P_z2)
    #start EM
    LogLikeiHood=sum([np.log2(P_z1*gaus1.pdf(point)+P_z2*gaus2.pdf(point)) for point in data_point])
    print("Log-Lokeihood is: ",LogLikeiHood)
    Logarray=np.array(LogLikeiHood)
    delta1=1
    while(delta1>epsilon):
        #E stage
        W=np.zeros((len(data_point),2))
        for j,z in enumerate(P_z):
            for i,point in enumerate(data_point):
                mone=z*(gaus[j].pdf(point))
                mechane=P_z[0]*(gaus[0].pdf(point))+P_z[1]*(gaus[1].pdf(point))
                W[i][j]=mone/mechane
        #M stage

        for j in range(len(P_z)):
            sumOfWeigth=np.sum(W[:,j])
            P_z[j]=np.mean(W[:,j])
            center[j][0]=np.dot(W[:,j],data_point[:,0])/sumOfWeigth
            center[j][1]=np.dot(W[:,j],data_point[:,1])/sumOfWeigth
            matrix=sum([W[i][j]*np.array(point-center[j]).reshape(1,2)*np.array(point-center[j]).reshape(2,1)
                           for i,point in enumerate(data_point)])
            cov[j]=matrix/sumOfWeigth
            gaus[j]=multivariate_normal(center[j],cov[j])
        newLogLikeiHood = sum([np.log2(P_z[0]*gaus[0].pdf(point)+P_z[1]*gaus[1].pdf(point)) for point in data_point])
        delta1=newLogLikeiHood-LogLikeiHood
        LogLikeiHood=newLogLikeiHood
        iteration+=1
        Logarray=np.append(Logarray,LogLikeiHood)
        print("iteration ",iteration,"Log-Lokeihood is: ",LogLikeiHood)
    print("Pz is: \n", P_z)
    print("Mean is: \n", center)
    print("Cov is: \n", cov)
    plt.pyplot.title("Log Likelihood of EM")
    plt.pyplot.plot(range(len(Logarray)), Logarray)
    plt.pyplot.xlabel("Iteration")
    plt.pyplot.ylabel("Loklihood")
    plt.pyplot.show()
    group1=np.array([])
    group2=np.array([])
    for i in range(len(W)):
        if(W[i][0]>W[i][1]):
            group1=np.append(group1,data_point[i])
        else:
            group2=np.append(group2,data_point[i])
    group1=group1.reshape(len(group1)//2,2)
    group2=group2.reshape(len(group2) // 2, 2)
    plt.pyplot.scatter(group1[:, 0], group1[:, 1], c="red", label="Group1")
    plt.pyplot.scatter(group2[:, 0], group2[:, 1], c="blue", label="Group2")
    plt.pyplot.title("EM, after run")
    plt.pyplot.legend()
    plt.pyplot.show()
    return 1

EM(0.01)








