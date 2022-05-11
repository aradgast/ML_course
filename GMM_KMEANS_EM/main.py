import numpy as np
import numpy.ma

import varbiels as vars
import matplotlib.pyplot as plt
from scipy import stats as sts
import timeit




def plot_gauss(mu, cov, color=None):
    N = 200
    X = np.linspace(-4, 4, N)
    Y = np.linspace(-4, 4, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    rv = sts.multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    plt.contour(X, Y, Z, colors=color, linewidths=2)


def normal_point(mean, cov):
    point = np.random.multivariate_normal(mean, cov)
    return [point[0], point[1]]


def get_random_psd(n):
    psd = np.random.rand(n, n)
    return np.tril(psd) + np.tril(psd, -1).T + n * np.eye(n)


def K_means(K, data, show=False):
    """steps in the algorithm:
    1. init k centroids
    2. repeat two steps until convergence:
        a. calc the distance between each data point to each centroid
        b. assign each data point to tne the nearest centroid
        """
    positions = np.random.choice(len(data), K, replace=False)
    centroids = []
    for pos in positions:
        centroids.append(data[pos])
    eps = 10 ** -6
    iter = 1
    while True:
        cluster = np.zeros((K, 1), dtype='object')
        cluster = [[0, 0] for elem in cluster]
        new_centroids = []
        for point in data:
            tmp_dis_vec = np.linalg.norm(np.array(point) - np.array(centroids), axis=1)
            clus = np.argmin(tmp_dis_vec)
            if cluster[clus] == [0, 0]:
                cluster[clus] = [point]
            else:
                cluster[clus].append(point)

        for cntr in cluster:
            t = np.transpose(cntr)
            x, y = np.average(t[0]), np.average(t[1])
            new_centroids.append([x, y])
        dif = np.linalg.norm(np.array(centroids) - np.array(new_centroids))
        if show:
            plt.figure(figsize=(10,8))
            colors = ['r', 'b', 'g', 'y']
            plt.scatter(np.transpose(new_centroids)[0], np.transpose(new_centroids)[1], s=200, c='lime')
            leg = ['centers']
            for k in range(K):
                p = np.transpose(cluster[k])
                plt.scatter(p[0], p[1], c=colors[k % 4])
                leg.append(f'cluster {k}')
            plt.title(f'Kmean - N. iteration {iter}')
            plt.legend(leg)
            plt.savefig(f'Kmean - N. iteration {iter}.jpeg')
            # plt.show()
        iter += 1
        if dif < eps:
            break
        else:
            centroids = new_centroids

    prob = [len(cluster[i])/len(data) for i in range(K)]
    var = np.array([np.zeros((2,2)) for i in range(K)])
    for i in range(K):
        z = np.array(cluster[i])-new_centroids[i]
        N = len(z)
        var[i] = np.matmul(np.transpose(z),z)/N


    return prob, new_centroids, var


def init_params_EM(dim, K, data, isKmeansinit=True):
    if isKmeansinit:
        phi, means, cov = K_means(K, data)
    else:
        phi = np.random.uniform(0, 1, size=(K, 1))
        phi = phi / sum(phi)
        means = [np.random.normal(0, 1, size=(dim,)) for k in range(K)]
        cov = [get_random_psd(dim) for k in range(K)]

    return phi, means, cov


def E_stpe(K, data, phi, gaus):
    # start = timeit.default_timer()
    w = np.zeros((K, len(data)))
    for i in range(len(data)):
        for j in range(K):
            numerator = phi[j] * gaus[j].pdf(data[i])
            w[j, i] = numerator
        w[:, i] /= sum(w[:, i])
    # print('E step time : ', timeit.default_timer()-start)
    return w


def M_step(data, w, K):
    # start = timeit.default_timer()
    N = len(data)
    new_phi = []
    new_mu = []
    new_cov = []
    for j in range(K):
        new_phi.append(np.average(w[j]))
    for j in range(K):
        numerator = sum([np.array(w[j, i]) * np.array(data[i]) for i in range(N)])
        denomater = sum(w[j])
        new_mu.append(numerator / denomater)
    for j in range(K):
        sub = np.subtract(data, new_mu[j])
        mat = [np.matmul(np.transpose([vec]), [vec]) for vec in sub]
        wi_mat = sum([w[j, i] * mat[i] for i in range(N)])
        new_cov.append(wi_mat / sum(w[j]))
    new_gaus = []
    for j in range(K):
        new_gaus.append(sts.multivariate_normal(new_mu[j], new_cov[j]))
    # print('M step time : ', timeit.default_timer()-start)
    return new_phi, new_mu, new_cov, new_gaus


def calc_l(data, phi, gaus):
    # start = timeit.default_timer()
    p = np.sum([np.log2(sum([phi[i] * gaus[i].pdf(x) for i in range(len(phi))])) for x in data])
    # print('ploss step time : ', timeit.default_timer()-start)
    return p


def EM_algo(K, data, isKmeansinit=True):
    """very similar to the Kmeans algorithm but with 'soft desiccation' - instead clustering each point to a certain
    distribution we calc the probability for each point to be in each possible distribution and then calc the new
    means and covs. """
    # init method
    eps = 10 ** -3
    phi, mu, cov = init_params_EM(2, K, data, isKmeansinit)
    gaus = []
    for j in range(K):
        gaus.append(sts.multivariate_normal(mu[j], cov[j]))
    # l_old = calc_l(data, phi, gaus_init)
    l_old = calc_l(data, phi, gaus)
    loss_vec = [l_old]
    iter = 0
    while True:
        # E-step
        w = E_stpe(K, data, phi, gaus)
        # M-step
        new_phi, new_mu, new_cov, new_gaus = M_step(data, w, K)
        l_new = calc_l(data, phi, new_gaus)
        loss_vec.append(l_new)
        if abs(l_new - l_old) < eps:
            break
        else:
            phi, mu, cov, gaus = new_phi, new_mu, new_cov, new_gaus
            l_old = l_new
            iter += 1
            # print("iter : ", iter)

    return new_phi, new_mu, new_cov, iter, loss_vec, w


if __name__ == '__main__':
    N = 1000
    mu1 = np.array([-1, -1])
    mu2 = np.array([1, 1])
    sig1 = np.array([[0.8, 0], [0, 0.8]])
    sig2 = np.array([[0.75, -0.2], [-0.2, 0.6]])
    p_z1 = 0.7
    gaus_dict = {0: [mu1, sig1], 1: [mu2, sig2]}
    data1 = []
    for i in range(N):
        z = np.random.binomial(1, 1-p_z1)
        data1.append(normal_point(gaus_dict[z][0], gaus_dict[z][1]) + [z])
    a = np.transpose(data1)
    plt.figure(figsize=(11,8))
    plt.scatter(a[0], a[1], c=a[2])
    plt.title('Data Generation - 1000 points')
    plt.savefig('Data Generation - 1000 points.jpeg')
    # plt.show()
    print('###################################################################################')
    ###################################################################################################################
    N = 50
    data2 = []
    for i in range(N):
        z = np.random.binomial(1, 1-p_z1)
        data2.append(normal_point(gaus_dict[z][0], gaus_dict[z][1]))
    b = np.transpose(data2)
    plt.figure(figsize=(11,8))
    plt.scatter(b[0], b[1])
    plt.title('50 data points')
    plt.savefig('50 data points.jpeg')
    # plt.show()
    print('###################################################################################')
    ###################################################################################################################
    print('Kmeans algorithm for K=2')
    c, cc, ccc = K_means(2, data2, show=True)
    print(f'the means we get are with kmeans: \n{cc[0]}\n{cc[1]}')
    ###################################################################################################################
    N = 10000
    data3 = []
    c1 = []
    for i in range(N):
        z = np.random.binomial(1, 1-p_z1)
        data3.append(normal_point(gaus_dict[z][0], gaus_dict[z][1]))
        c1.append(z)
    a2 = np.transpose(data3)
    plt.figure(figsize=(11,8))
    plt.scatter(a2[0], a2[1])
    plt.suptitle(f'Data Generation - 10,000 points', fontsize=16)
    plt.title(f'1: {gaus_dict[0][0]}  \n {gaus_dict[0][1]}', fontsize=12, loc='left', c='red')
    plt.title(f'2: {gaus_dict[1][0]} \n {gaus_dict[1][1]}', fontsize=12, loc='right', c='blue')
    plot_gauss(gaus_dict[0][0], gaus_dict[0][1], 'red')
    plot_gauss(gaus_dict[1][0], gaus_dict[1][1], 'blue')
    plt.legend()
    plt.grid()
    plt.savefig(f'Data Generation - 10,000 points.jpeg')
    # plt.show()
    ##########################################
    print('EM algorithm for k=2, randomly init')
    phi, mu, cov, iter, loss, w = EM_algo(2, data3, False)
    print('phi = \n', phi, '\nmu = \n', mu, '\ncov = \n', cov, '\nnum of iterations = \n', iter)
    plt.figure(figsize=(11,8))
    plt.plot([i for i in range(len(loss))], loss)
    plt.title('log-likelihood function,k=2, init randomly')
    plt.xlabel('number of iterations')
    plt.ylabel('log-loss')
    plt.grid()
    plt.savefig('log-likelihood function,k=2, init randomly.jpeg')
    # plt.show()
    plt.figure(figsize=(11,8))
    plt.legend(['cluster 0 ', 'cluster 1'])
    plt.scatter(a2[0], a2[1])
    plot_gauss(mu[0], cov[0], 'red')
    plot_gauss(mu[1], cov[1], 'blue')
    plt.suptitle('data clustered after EM,k=2, init randomly', fontsize=16)
    plt.title(f'1: {mu[0]}  \n {cov[0]}', fontsize=12, loc='left', c='red')
    plt.title(f'2: {mu[1]} \n {cov[1]}', fontsize=12, loc='right', c='blue')
    plt.grid()
    plt.savefig('data clustered after EM,k=2, init randomly.jpeg')
    # plt.show()
    print('###################################################################################')
    ################################################################################
    print('EM algorithm for k=2, Kmeans init')
    phi, mu, cov, iter, loss, w = EM_algo(2, data3, True)
    print('phi = \n', phi, '\nmu = \n', mu, '\ncov = \n', cov, '\nnum of iterations = \n', iter)
    plt.figure(figsize=(11,8))
    plt.plot([i for i in range(len(loss))], loss)
    plt.title('log-likelihood function,k=2, init Kmeans')
    plt.xlabel('number of iterations')
    plt.ylabel('log-loss')
    plt.grid()
    plt.savefig('log-likelihood function,k=2, init Kmeans.jpeg')
    # plt.show()
    plt.figure(figsize=(11,8))
    plt.scatter(a2[0], a2[1])
    plot_gauss(mu[0], cov[0], 'red')
    plot_gauss(mu[1], cov[1], 'blue')
    plt.suptitle('data clustered after EM,k=2, Kmeans init', fontsize=16)
    plt.title(f'1: {mu[0]}  \n {cov[0]}', fontsize=12, loc='left', c='red')
    plt.title(f'2: {mu[1]} \n {cov[1]}', fontsize=12, loc='right', c='blue')
    plt.grid()
    plt.savefig('data clustered after EM,k=2, Kmeans init.jpeg')
    # plt.show()
    print('###################################################################################')

    ###################################################################################
    print('EM algorithm for k=3, randomly init')
    phi, mu, cov, iter, loss, w = EM_algo(3, data3, False)
    print('phi = \n', phi, '\nmu = \n', mu, '\ncov = \n', cov, '\nnum of iterations = \n', iter)
    plt.figure(figsize=(11,8))
    plt.plot([i for i in range(len(loss))], loss)
    plt.title('log-likelihood function,k=3, init randomly')
    plt.xlabel('number of iterations')
    plt.ylabel('log-loss')
    plt.grid()
    plt.savefig('log-likelihood function,k=3, init randomly.jpeg')
    # plt.show()
    plt.figure(figsize=(11,8))
    plt.scatter(a2[0], a2[1])
    plot_gauss(mu[0], cov[0], 'red')
    plot_gauss(mu[1], cov[1], 'blue')
    plot_gauss(mu[2], cov[2], 'green')
    plt.suptitle('data clustered after EM,k=3, random init', fontsize=16)
    plt.title(f'1: {mu[0]}  \n {cov[0]}', fontsize=12, loc='left', c='red')
    plt.title(f'2: {mu[1]} \n {cov[1]}', fontsize=12, loc='center', c='blue')
    plt.title(f'3: {mu[2]} \n {cov[2]}', fontsize=12, loc='right', c='green')
    plt.grid()
    plt.savefig('data clustered after EM,k=3, random init.jpeg')
    # plt.show()
    print('###################################################################################')

    ###########################################################################################################
    print('EM algorithm for k=3, kmeans init')
    phi, mu, cov, iter, loss, w = EM_algo(3, data3, True)
    print('phi = \n', phi, '\nmu = \n', mu, '\ncov = \n', cov, '\nnum of iterations = \n', iter)
    plt.figure(figsize=(11,8))
    plt.plot([i for i in range(len(loss))], loss)
    plt.title('log-likelihood function,k=3, init Kmeans')
    plt.xlabel('number of iterations')
    plt.ylabel('log-loss')
    plt.grid()
    plt.savefig('log-likelihood function,k=3, init Kmeans.jpeg')
    # plt.show()
    plt.figure(figsize=(11,8))
    plt.scatter(a2[0], a2[1])
    plot_gauss(mu[0], cov[0], 'red')
    plot_gauss(mu[1], cov[1], 'blue')
    plot_gauss(mu[2], cov[2], 'green')
    plt.suptitle('data clustered after EM,k=3, Kmeans init', fontsize=16)
    plt.title(f'1: {mu[0]}  \n {cov[0]}', fontsize=12, loc='left', c='red')
    plt.title(f'2: {mu[1]} \n {cov[1]}', fontsize=12, loc='center', c='blue')
    plt.title(f'3: {mu[2]} \n {cov[2]}', fontsize=12, loc='right', c='green')
    plt.grid()
    plt.savefig('data clustered after EM,k=3, Kmeans init.jpeg')
    # plt.show()

    ###################################################################################

    print('done')
