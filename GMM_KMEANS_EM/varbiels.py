import numpy as np
def gaus_point(mean, cov):
    point = np.random.multivariate_normal(mean, cov)
    return [point[0], point[1]]
if __name__ == '__main__':
    N = 1000
    mu1 = np.array([-1, -1])
    mu2 = np.array([1, 1])
    sig1 = np.array([[0.8, 0], [0, 0.8]])
    sig2 = np.array([[0.75, -0.2], [-0.2, 0.6]])
    p_z1 = 0.7
    gaus_dict = {1: [mu1, sig1], 2: [mu2, sig2]}
    data1 = []
    for i in range(N):
        z = np.random.binomial(1, p_z1) + 1
        data1.append(gaus_point(gaus_dict[z][0], gaus_dict[z][1]))
    a = np.transpose(data1)

    N = 50
    data2 = []
    for i in range(N):
        z = np.random.binomial(1, p_z1) + 1
        data2.append(gaus_point(gaus_dict[z][0], gaus_dict[z][1]))
    b = np.transpose(data2)