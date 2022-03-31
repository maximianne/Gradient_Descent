# Name: Maxie Castaneda
# COMP 347 - Machine Learning
# HW No. 3

# Libraries
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Problem 1 - Gradient Descent Using Athens Temperature Data
# ------------------------------------------------------------------------------
# For this problem you will be implementing various forms of gradient descent  
# using the Athens temperature data.  Feel free to copy over any functions you 
# wrote in HW #2 for this.  WARNING: In order to get gradient descent to work
# well, I highly recommend rewriting your cost function so that you are dividing
# by N (i.e. the number of data points there are).  Carry this over into all 
# corresponding expression where this would appear (the derivative being one of them).

# functions from hw 1
def A_mat(x, deg):
    """Create the matrix A part of the least squares problem.
       x: vector of input datas.
       deg: degree of the polynomial fit."""
    A = np.zeros((len(x), deg + 1))  # initialize a matrix full of zeros
    count = deg
    for i in range(deg + 1):
        for j in range(len(x)):
            val = x[j]
            A[j, i] = val ** count
        count -= 1
    return A


def LLS_Solve(x, y, deg):
    """Find the vector w that solves the least squares regression.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.
       w = (A'A)-1 A'Y """
    A = A_mat(x, deg)
    AT = A.transpose()
    ATA = np.matmul(AT, A)
    ATAInv = np.linalg.inv(ATA)
    ATY = np.matmul(AT, y)
    w = np.matmul(ATAInv, ATY)
    return w / len(x)


def cost_function(x, w, y, deg):
    A = A_mat(x, deg)
    costVal = np.linalg.norm(np.matmul(A, w) - y)
    return costVal


# 1a. Fill out the function for the derivative of the least-squares cost function:
def LLS_deriv(x, y, w, deg):
    """Computes the derivative of the least squares cost function with input
    data x, output data y, coefficient vector w, and deg (the degree of the
    least squares model)."""
    # the LLS cost function : ||AW - y||^2
    # derivative of LLS cost function = 2A'Aw - 2A'y
    # 2A'(Aw-y)
    A = A_mat(x, deg)
    diff = np.matmul(A, w) - y
    # print('diff.shape', diff.shape)
    return (2 * np.matmul(np.transpose(A), diff)) / len(x)


def GD(x, y, betas, k, epsilon):
    c = cost_function(x, betas, y, 1)
    countGD = 0
    # print('LLS_Solve:', LLS_Solve(X, Y, 1))
    d = np.linalg.norm(LLS_deriv(x, y, betas, 1))
    c_hist_gd = []
    d_hist_gd = []
    while np.linalg.norm(d) >= k:
        d = LLS_deriv(x, y, betas, 1)
        d_hist_gd.append(np.linalg.norm(d))
        c = cost_function(x, betas, y, 1)
        c_hist_gd.append(c)
        betas = betas - np.multiply(epsilon, (LLS_deriv(x, y, betas, 1)))
        countGD += 1
    return w, c_hist_gd, d_hist_gd


def minibatch(x, y, w, k, batch_size, beta, epsilon):
    d = np.linalg.norm(LLS_deriv(x, y, w, 1))
    ci = 0

    batches_x = []
    batches_y = []

    c_hist_m = []
    d_hist_m = []

    for i in range(int(x.shape[0] / batch_size)):
        mini_x = x[i * batch_size:(i + 1) * batch_size]
        mini_y = y[i * batch_size:(i + 1) * batch_size]
        mini_x = np.array(mini_x)
        mini_y = np.array(mini_y)
        batches_x.append(mini_x)
        batches_y.append(mini_y)

    while np.linalg.norm(d) >= k:
        for i in range(0, len(batches_x)):
            d = LLS_deriv(batches_x[i], batches_y[i], w, 1)
            d_hist_m.append(np.linalg.norm(d))
            if np.linalg.norm(d) <= k:
                break
            c = cost_function(x, w, y, 1)
            c_hist_m.append(c)
            M = np.linalg.norm(d) ** 2
            T = beta * M
            new_eps = epsilon
            while cost_function(x, w - new_eps * d, y, 1) > cost_function(x, w, y, 1) + T:
                new_eps *= 0.9
            w = w - np.multiply(new_eps, d)
            ci += 1
    return w, c_hist_m, d_hist_m, ci


def stochiastic_GD(x, y, betas, k):
    d = np.linalg.norm(LLS_deriv(x, y, betas, 1))
    c_hist1 = []
    d_hist1 = []
    ci = 0
    while np.linalg.norm(d) >= k:
        df = pd.DataFrame([x, y]).transpose()
        mini = df.sample(n=1)
        mini_x = np.array([mini.iat[0, 0]])
        mini_y = np.array([mini.iat[0, 1]])
        epsilon = 0.001
        beta = 0.000000001
        d = LLS_deriv(mini_x, mini_y, betas, 1)
        c = cost_function(mini_x, betas, mini_y, 1)
        c_hist1.append(c)
        d_hist1.append(np.linalg.norm(d))
        M = np.linalg.norm(d) ** 2
        t = beta * M
        betas = betas - np.multiply(epsilon, d)
        ci += 1
    c_hist1 = np.array(c_hist1)
    d_hist1 = np.array(d_hist1)
    return betas, c_hist1, d_hist1, ci


# 2a. Fill out the function for the soft-thresholding operator S_lambda as discussed
#     in lecture:
def soft_thresh(v, lam):
    """Perform the soft-thresholding operation of the vector v using parameter lam."""
    size = len(v)
    ans = np.zeros(size)
    for i in range(size):
        if v[i] > lam:
            ans[i] = v[i] - lam
        elif -lam <= v[i] <= lam:
            ans[i] = 0
        else:
            ans[i] = v[i] + lam
    return ans


def cost_functionLASSO(x, y, w, deg, lam):
    A = A_mat(x, deg)
    costVal = np.linalg.norm(np.matmul(A, w) - y)
    wn = np.linalg.norm(w)
    lam_w = lam * wn
    cost = costVal + lam_w
    return cost


def lasso_opt(x, y, w, lam, k):
    w_euc = []
    w_ord1 = []
    w_list = []
    d = np.linalg.norm(LLS_deriv(x, y, w, 1))
    deriv = LLS_deriv(x, y, w, 1)
    iteration = 0
    while np.linalg.norm(d) >= k and iteration < 20000:
        pre = w - np.multiply(lam, deriv)
        w = soft_thresh(pre, lam)
        cost = cost_functionLASSO(x, y, w, 1, lam)
        deriv = LLS_deriv(X, Y, w, 1)
        d = np.linalg.norm(LLS_deriv(x, y, w, 1))
        print("cost: ", cost)
        print("W:                  ", w)
        print('deriv     :', deriv)
        print('deriv norm:', d)
        d = np.linalg.norm(LLS_deriv(x, y, w, 1))
        iteration += 1
    return w, w_euc, w_ord1, w_list


if __name__ == "__main__":
    data = pd.read_csv('machine learning/athens_ww2_weather.csv')
    X = data['MinTemp']
    Y = data['MaxTemp']
    # 1b. Implement gradient descent as a means of optimizing the least squares cost
    #     function.  Your method should include the following:
    #       a. initial vector w that you are optimizing
    w = [100, 10]
    #       b. a tolerance K signifying the acceptable derivative norm for stopping
    #          the descent method
    K = 5

    #       c. initial derivative vector D (initialization at least for the sake of
    #          starting the loop)
    d_0 = [-1, 1]
    #       d. an empty list called d_hist which captures the size (i.e. norm) of your
    #          derivative vector at each iteration,
    d_hist_b = []

    #       e. an empty list called c_hist which captures the cost (i.e. value of
    #          the cost function) at each iteration,
    c_hist_b = []

    # gradient descent
    # while ||DERIV (wi) || >- K
    # compute: wi+1 = wi - e(DERIV (wi))
    # where e is the learning rate
    e = 0.001

    w, hist_c, hist_d = GD(X, Y, w, K, e)
   
    #       f. implement backtracking line search as part of your steepest descent
    #           algorithm.  You can implement this on your own if you're feeling
    #           cavalier, or if you'd like here's a snippet of what I used in mine:
    #
    #                eps = 1
    #                m = LA.norm(D)**2
    #                t = 0.5*m
    #                while LLS_func(a_min, a_max, w - eps*D, 1) > LLS_func(a_min, a_max, w, 1) - eps*t:
    #                    eps *= 0.9

    #       Plot curves showing the derivative size (i.e. d_hist) and cost (i.e. c_hist)
    #       with respect to the number of iterations.
    countBTLS = 0
    deriv = np.linalg.norm(LLS_deriv(X, Y, w, 1))
    while np.linalg.norm(deriv) >= K:
        eps = 0.01
        deriv = LLS_deriv(X, Y, w, 1)
        cost = cost_function(X, w, Y, 1)
        c_hist_b.append(cost)
        d_hist_b.append(np.linalg.norm(deriv))
        m = np.linalg.norm(deriv) ** 2
        t = 0.5 * m
        while cost_function(X, w - eps * deriv, Y, 1) > cost_function(X, w, Y, 1):
            eps *= 0.9
        w = w - np.multiply(eps, deriv)
        countBTLS += 1

    c_hist_b = np.array(c_hist_b)
    d_hist_b = np.array(d_hist_b)

    plt.title('Cost and Gradient Descent with Backtracking Line Search')
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Cost', fontsize=8)
    plt.plot(c_hist_b, label='Cost')
    plt.plot(d_hist_b, label='Derivative')
    plt.legend()
    plt.show()

    # 1c. Repeat part 1b, but now implement mini-batch gradient descent by randomizing
    #     the data points used in each iteration.  Perform mini-batch descent for batches
    #     of size 5, 10, 25, and 50 data points.  For each one, plot the curves
    #     for d_hist and c_hist.  Plot these all on one graph showing the relationship
    #     between batch size and convergence speed (at least for the least squares
    #     problem).  Feel free to adjust the transparency of your curves so that
    #     they are easily distinguishable.
    w = [100, 10]
    #       b. a tolerance K signifying the acceptable derivative norm for stopping
    #          the descent method
    K = 5
    # 5
    w5, c_hist_mini5, d_hist_mini5, countIteration = minibatch(X, Y, w, k=5, batch_size=5, beta=0.000001, epsilon=.5)
    print(w5)

    # 10
    w10, c_hist_mini10, d_hist_mini10, countIteration = minibatch(X, Y, w, k=2.5, batch_size=10, beta=0.0000001,
                                                                  epsilon=.05)
    print(w10)

    # 25
    w25, c_hist_mini25, d_hist_mini25, countIteration = minibatch(X, Y, w, k=3, batch_size=25, beta=0.000001,
                                                                  epsilon=0.01)
    print(w25)

    # 50
    w50, c_hist_mini50, d_hist_mini50, countIteration = minibatch(X, Y, w, k=3, batch_size=50, beta=0.000001,
                                                                  epsilon=0.001)
    print(w50)

    plt.title('Mini-batch GD Cost of different sizes')
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Cost', fontsize=8)
    plt.plot(c_hist_mini5, label='Mini = 5')
    plt.plot(c_hist_mini10, label='Mini = 10')
    plt.plot(c_hist_mini25, label='Mini = 25')
    plt.plot(c_hist_mini50, label='Mini = 50')
    plt.legend()
    plt.show()

    plt.title('Mini-batch GD Derivative Size of different sizes')
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Cost', fontsize=8)
    plt.plot(d_hist_mini5, label='Mini = 5')
    plt.plot(d_hist_mini10, label='Mini = 10')
    plt.plot(d_hist_mini25, label='Mini = 25')
    plt.plot(d_hist_mini50, label='Mini = 50')
    plt.legend()
    plt.show()

    # 1d. Repeat 1b, but now implement stochastic gradient descent.  Plot the curves
    #     for d_hist and c_hist.  WARNING: There is a strong possibility that your
    #     cost and derivative definitions may not compute the values correctly for the 1-dimensional case.  If needed,
    #     make sure that you adjust these functions to accommodate a single data point.

    (w, c_hist, d_hist, iteration) = stochiastic_GD(Y, X, w, K)
    print(w)
    plt.title('Stochastic GD Derivative and Cost size')
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Cost', fontsize=8)
    plt.plot(c_hist, label='cost')
    plt.plot(d_hist, label='derivative size')
    plt.legend()
    plt.show()

    # 1e. Aggregate your curves for batch, mini-batch, and stochastic descent methods
    #     into one final graph so that a full comparison between all methods can be
    #     observed.  Make sure your legend clearly indicates the results of each
    #     method.  Adjust the transparency of the curves as needed.

    plt.title('Derivative Size Comparison of Various Methods')
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Cost', fontsize=8)
    plt.plot(d_hist_b, label='Batch')
    plt.plot(d_hist_mini5, label='Mini = 5')
    plt.plot(d_hist_mini10, label='Mini = 10')
    plt.plot(d_hist_mini25, label='Mini = 25')
    plt.plot(d_hist_mini50, label='Mini = 50')
    plt.plot(d_hist, label='Stochastic')
    plt.legend()
    plt.show()

    # Problem 2 - LASSO Regularization
    # ------------------------------------------------------------------------------
    # For this problem you will be implementing LASSO regression on the Yosemite data.

    # 2b. Using 5 years of the Yosemite data, perform LASSO regression with the values
    #     of lam ranging from 0.25 up to 5, spacing them in increments of 0.25.
    #     Specifically do this for a cubic model of the Yosemite data.  In doing this
    #     save each of your optimal parameter vectors w to a list as well as solving
    #     for the exact solution for the least squares problem.  Make the following
    #     graphs:

    w = [90, 100]
    # exact
    lam = 0
    wL, w_euc, w_ord, wList = lasso_opt(X, Y, w, lam, 5)

    lam = .1 / len(X)
    wL1, w_euc1, w_ord1, w1 = lasso_opt(X, Y, w, lam, 5)

    lam = .2 / len(X)
    wL2, w_euc2, w_ord2, w2 = lasso_opt(X, Y, w, lam, 5)

    lam = .3 / len(X)
    wL3, w_euc3, w_ord3, w3 = lasso_opt(X, Y, w, lam, 5)

    lam = .4 / len(X)
    wL4, w_euc4, w_ord4, w4 = lasso_opt(X, Y, w, lam, 5)

    lam = .5 / len(X)
    wL5, w_euc5, w_ord5, w5 = lasso_opt(X, Y, w, lam, 5)

    lam = .6 / len(X)
    wL6, w_euc6, w_ord6, w6 = lasso_opt(X, Y, w, lam, 5)

    lam = .7 / len(X)
    wL7, w_euc7, w_ord7, w7 = lasso_opt(X, Y, w, lam, 5)

    lam = .8 / len(X)
    wL8, w_euc8, w_ord8, w8 = lasso_opt(X, Y, w, lam, 5)

    lam = .9 / len(X)
    wL9, w_euc9, w_ord9, w9 = lasso_opt(X, Y, w, lam, 5)

    lam = 1 / len(X)
    wL10, w_euc10, w_ord10, w10 = lasso_opt(X, Y, w, lam, 5)

    lam = 1.1 / len(X)
    wL11, w_euc11, w_ord11, w11 = lasso_opt(X, Y, w, lam, 5)

    lam = 1.2 / len(X)
    wL12, w_euc12, w_ord12, w12 = lasso_opt(X, Y, w, lam, 5)

    lam = 1.3 / len(X)
    wL13, w_euc13, w_ord13, w13 = lasso_opt(X, Y, w, lam, 5)

    lam = 1.4 / len(X)
    wL14, w_euc14, w_ord14, w14 = lasso_opt(X, Y, w, lam, 5)

    lam = 1.5 / len(X)
    wL15, w_euc15, w_ord15, w15 = lasso_opt(X, Y, w, lam, 5)

    #       a. Make a graph of the l^2 norms (i.e. Euclidean) and l^1 norms of the
    #          optimal parameter vectors w as a function of the coefficient lam.
    #          Interpret lam = 0 as the exact solution.  One can find the 1-norm of
    #          a vector using LA.norm(w, ord = 1)

    w_euc = [np.linalg.norm(wL), np.linalg.norm(wL1), np.linalg.norm(wL2), np.linalg.norm(wL3), np.linalg.norm(wL4),
             np.linalg.norm(wL5), np.linalg.norm(wL6), np.linalg.norm(wL7), np.linalg.norm(wL8), np.linalg.norm(wL9),
             np.linalg.norm(wL10), np.linalg.norm(wL11), np.linalg.norm(wL12), np.linalg.norm(wL13),
             np.linalg.norm(wL14),
             np.linalg.norm(wL15)]

    w_eucO1 = [np.linalg.norm(wL, ord=1), np.linalg.norm(wL1, ord=1), np.linalg.norm(wL2, ord=1),
               np.linalg.norm(wL3, ord=1),
               np.linalg.norm(wL4, ord=1), np.linalg.norm(wL5, ord=1), np.linalg.norm(wL6, ord=1),
               np.linalg.norm(wL7, ord=1),
               np.linalg.norm(wL8, ord=1), np.linalg.norm(wL9, ord=1), np.linalg.norm(wL10, ord=1),
               np.linalg.norm(wL11, ord=1),
               np.linalg.norm(wL12, ord=1), np.linalg.norm(wL13, ord=1), np.linalg.norm(wL14, ord=1),
               np.linalg.norm(wL15, ord=1)]

    x_m = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    lam_vals = ["0", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9", "1", "1.1", "1.2",
                "1.3", "1.4", "1.5"]

    plt.title('Euclidean Norm and Norm O =1 for optimal parameters W')
    plt.xlabel('Lambda', fontsize=10)
    plt.ylabel('Size', fontsize=8)
    plt.xticks(x_m, lam_vals)
    plt.plot(x_m, w_euc, label='Euclidean')
    plt.plot(x_m, w_eucO1, label='Norm Ord = 1')
    plt.legend()
    plt.show()

    #       b. For each coefficient in the cubic model (i.e. there are 4 of these),
    #           make a separate plot of the absolute value of the coefficient as a
    #           function of the parameter lam (again, lam = 0 should be the exact
    #           solution to the original least squares problem).  Is there a
    #           discernible trend of the sizes of our entries for increasing values
    #           of lam?
    ''' as Lam gets larger, the norms and then the coefficients get smallers'''

    coef_1 = [np.absolute(wL[0]), np.absolute(wL1[0]), np.absolute(wL2[0]), np.absolute(wL3[0]), np.absolute(wL4[0]),
              np.absolute(wL5[0]), np.absolute(wL6[0]), np.absolute(wL7[0]), np.absolute(wL8[0]), np.absolute(wL9[0]),
              np.absolute(wL10[0]), np.absolute(wL11[0]), np.absolute(wL12[0]), np.absolute(wL13[0]),
              np.absolute(wL14[0]),
              np.absolute(wL15[0])]
    coef_2 = [np.absolute(wL[1]), np.absolute(wL1[1]), np.absolute(wL2[1]), np.absolute(wL3[1]), np.absolute(wL4[1]),
              np.absolute(wL5[1]), np.absolute(wL6[1]), np.absolute(wL7[1]), np.absolute(wL8[1]), np.absolute(wL9[1]),
              np.absolute(wL10[1]), np.absolute(wL11[1]), np.absolute(wL12[1]), np.absolute(wL13[1]),
              np.absolute(wL14[1]),
              np.absolute(wL15[1])]

    plt.title('Euclidean Norm and Norm O =1 for optimal parameters W')
    plt.xlabel('Lambda', fontsize=10)
    plt.ylabel('size', fontsize=8)
    plt.xticks(x_m, lam_vals)
    plt.plot(x_m, coef_1, label='w_0')
    plt.plot(x_m, coef_2, label='w_1')
    plt.legend()
    plt.show()

# Friendly Reminder: for LASSO regression you don't need backtracking line search.
# In essence the parameter lam serves as our step size.
