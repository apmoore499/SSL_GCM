#!/usr/bin/env python
# encoding: utf-8


import torch
import numpy as np
from functools import reduce
min_var_est = 1e-8


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean

@torch.jit.script
def _mix_linear_kernel(X, Y):
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    K = torch.mm(Z, Z.t())

    return K[:m, :m], K[:m, m:], K[m:, m:]


def mix_linear_kernel(X, Y):
    return _mix_linear_kernel(X, Y)


from typing import Dict,List
import torch

def _mix_rbf_kernel(X, Y, sigma_list):
    
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_kernel(X, Y, sigma_list):
    return _mix_rbf_kernel(X, Y, sigma_list)


def _mix_rbf_kernel_discrete(X, Y):
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()
    sigma = 0.01
    gamma = 1.0 / (2 * sigma ** 2)
    K = torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:]


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_linear_mmd2_joint(X, Y, X1, Y1, X2=None, Y2=None, sigma_list=None, biased=True):
    K_XX, K_XY, K_YY = _mix_linear_kernel(X, Y)
    K_XX1, K_XY1, K_YY1 = _mix_linear_kernel(X1, Y1)
    if X2 is None and Y2 is None:
        K_XX = K_XX * K_XX1
        K_YY = K_YY * K_YY1
        K_XY = K_XY * K_XY1
    else:
        K_XX2, K_XY2, K_YY2 = _mix_linear_kernel(X2, Y2)
        K_XX = K_XX * K_XX1 * K_XX2
        K_YY = K_YY * K_YY1 * K_YY2
        K_XY = K_XY * K_XY1 * K_XY2
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_joint(X, Y, X1, Y1, X2=None, Y2=None, X3=None, Y3=None, sigma_list=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    K_XX1, K_XY1, K_YY1 = _mix_linear_kernel(X1, Y1)
    if X2 is None and Y2 is None:
        K_XX = K_XX * K_XX1
        K_YY = K_YY * K_YY1
        K_XY = K_XY * K_XY1
    elif X3 is None and Y3 is None:
        K_XX2, K_XY2, K_YY2 = _mix_linear_kernel(X2, Y2)
        K_XX = K_XX * K_XX1 * K_XX2
        K_YY = K_YY * K_YY1 * K_YY2
        K_XY = K_XY * K_XY1 * K_XY2
    else:
        K_XX2, K_XY2, K_YY2 = _mix_linear_kernel(X2, Y2)
        K_XX3, K_XY3, K_YY3, d1 = _mix_rbf_kernel(X3, Y3, sigma_list)
        K_XX = K_XX * K_XX1 * K_XX2 * K_XX3
        K_YY = K_YY * K_YY1 * K_YY2 * K_YY3
        K_XY = K_XY * K_XY1 * K_XY2 * K_XY3
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)



#def MMD_multiple(X, Y, X1, Y1, X2=None, Y2=None, X3=None, Y3=None, sigma_list=None, sigma_list1=None, biased=True):
#computes MMD between multiple features and possible label - general solution to MMD implementation



def get_rbf_kernel_from_dict(**kwargs):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(kwargs['est'], kwargs['ground_truth'], kwargs['sigma_list'])
    retval={'K_XX':K_XX,
            'K_XY':K_XY,
            'K_YY':K_YY,
            'd':d}
    return(retval)

def get_linear_kernel_from_dict(**kwargs):
    K_XX, K_XY, K_YY = _mix_linear_kernel(kwargs['est'], kwargs['ground_truth'])
    retval={'K_XX':K_XX,
            'K_XY':K_XY,
            'K_YY':K_YY}
    return(retval)








# MMD estimate of several distributions (calculates them disjointly!!!)
# Archer 21_06_2022
# Will work if label_dict is empty also
# Will work for single feature

def MMD_multiple(feature_dict={},label_dict={},biased=True):
    #simple renaming for more compact
    fd=feature_dict
    ld=label_dict

    #calculate rbf kernel embeddings
    rbf_kernel_dict={var:get_rbf_kernel_from_dict(**fd[var]) for var in fd.keys()}

    #calculate linear kernel embeddings
    linear_kernel_dict={var:get_linear_kernel_from_dict(**ld[var]) for var in ld.keys()}

    #merge dicts
    all_kernel_dict={**rbf_kernel_dict,** linear_kernel_dict}

    # now pull out K_XX,K_YY,K_XY terms
    all_K_XX=[all_kernel_dict[v]['K_XX'] for v in all_kernel_dict.keys()]
    all_K_YY = [all_kernel_dict[v]['K_YY'] for v in all_kernel_dict.keys()]
    all_K_XY = [all_kernel_dict[v]['K_XY'] for v in all_kernel_dict.keys()]
    #multiply terms

    K_XX_prod = reduce((lambda x, y: x * y), all_K_XX)#torch.mul(*all_K_XX)
    K_YY_prod = reduce((lambda x, y: x * y), all_K_YY)
    K_XY_prod = reduce((lambda x, y: x * y), all_K_XY)

    #all done: return values

    return _mmd2(K_XX_prod, K_XY_prod, K_YY_prod, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_joint_regress(X, Y, X1, Y1, X2=None, Y2=None, X3=None, Y3=None, sigma_list=None, sigma_list1=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    K_XX1, K_XY1, K_YY1, d1 = _mix_rbf_kernel(X1, Y1, sigma_list1)
    if X2 is None and Y2 is None:
        K_XX = K_XX * K_XX1
        K_YY = K_YY * K_YY1
        K_XY = K_XY * K_XY1
    elif X3 is None and Y3 is None:
        K_XX2, K_XY2, K_YY2 = _mix_linear_kernel(X2, Y2)
        K_XX = K_XX * K_XX1 * K_XX2
        K_YY = K_YY * K_YY1 * K_YY2
        K_XY = K_XY * K_XY1 * K_XY2
    else:
        K_XX2, K_XY2, K_YY2 = _mix_linear_kernel(X2, Y2)
        K_XX3, K_XY3, K_YY3, d2 = _mix_rbf_kernel(X3, Y3, sigma_list)
        K_XX = K_XX * K_XX1 * K_XX2 * K_XX3
        K_YY = K_YY * K_YY1 * K_YY2 * K_YY3
        K_XY = K_XY * K_XY1 * K_XY2 * K_XY3
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


# def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
#     m = K_XX.size(0)  # assume X, Y are same shape
#
#     # Get the various sums of kernels that we'll use
#     # Kts drop the diagonal, but we don't need to compute them explicitly
#     if const_diagonal is not False:
#         diag_X = diag_Y = const_diagonal
#         # sum_diag_X = sum_diag_Y = m * const_diagonal
#     else:
#         diag_X = torch.diag(K_XX)  # (m,)
#         diag_Y = torch.diag(K_YY)  # (m,)
#         # sum_diag_X = torch.sum(diag_X)
#         # sum_diag_Y = torch.sum(diag_Y)
#
#     if biased:
#         Kt_XX_sums = K_XX.sum(dim=1)
#         Kt_YY_sums = K_YY.sum(dim=1)
#     else:
#         Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
#         Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
#     K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e
#
#     Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
#     Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
#     K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e
#
#     if biased:
#         # mmd2 = (Kt_XX_sum / (m * m)
#         #         + Kt_YY_sum / (m * m)
#         #         - 2.0 * K_XY_sum / (m * m))
#         mmd2 = K_XX - 2.0 * K_XY + K_YY
#         mmd2 = mmd2 / (m * m)
#         mmd2 = mmd2.sum().sum()
#     else:
#         mmd2 = (Kt_XX_sum / (m * (m - 1))
#                 + Kt_YY_sum / (m * (m - 1))
#                 - 2.0 * K_XY_sum / (m * m))
#
#     return mmd2


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=True):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal ** 2
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)  # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X  # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y  # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum = (K_XY ** 2).sum()  # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    var_est = (
        2.0 / (m ** 2 * (m - 1.0) ** 2) * (
        2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4.0 * m - 6.0) / (m ** 3 * (m - 1.0) ** 3) * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
        + 4.0 * (m - 2.0) / (m ** 3 * (m - 1.0) ** 2) * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0 * (m - 3.0) / (m ** 3 * (m - 1.0) ** 2) * (K_XY_2_sum) - (8 * m - 12) / (m ** 5 * (m - 1)) * K_XY_sum ** 2
        + 8.0 / (m ** 3 * (m - 1.0)) * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
    )
    return mmd2, var_est



# #custom compiled kern for if we need to do many mmd per run, speedup

# def _mix_linear_kernel(X, Y):
#     assert (X.size(0) == Y.size(0))
#     m = X.size(0)

#     Z = torch.cat((X, Y), 0)
#     K = torch.mm(Z, Z.t())

#     return K[:m, :m], K[:m, m:], K[m:, m:]



# class mix_rbf_kernel_class(torch.nn.Module):
    
#     def __init__(self,sigma_list):
#         super(mix_rbf_kernel_class, self).__init__()
#         self.sigma_list=sigma_list
        

#     def forward(self,X,Y):
#         assert (X.size(0) == Y.size(0))
#         m = X.size(0)
#         Z = torch.cat((X, Y), 0)
#         ZZT = torch.mm(Z, Z.t())
#         diag_ZZT = torch.diag(ZZT).unsqueeze(1)
#         Z_norm_sqr = diag_ZZT.expand_as(ZZT)
#         exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()
#         K=torch.sum(torch.exp(exponent[None,:,:].expand(self.sigma_list.shape[0],-1,-1) * -(1.0 / (2 * self.sigma_list ** 2))[:,None,None]), dim=0)
        
#         return K[:m, :m], K[:m, m:], K[m:, m:], len(self.sigma_list)


# class mix_rbf_mmd2_class(torch.nn.Module):
    
#     def __init__(self,sigma_list):
#         super(mix_rbf_mmd2_class, self).__init__()
#         self.rbf_kernel=mix_rbf_kernel_class(sigma_list=sigma_list)
        
#     def forward(self,X,Y):
#         K_XX, K_XY, K_YY, d = self.rbf_kernel(X, Y)
#         m = K_XX.size(0)    # assume X, Y are same shape

#         # # Get the various sums of kernels that we'll use
#         # # Kts drop the diagonal, but we don't need to compute them explicitly
#         # if const_diagonal is not False:
#         #     diag_X = diag_Y = const_diagonal
#         #     sum_diag_X = sum_diag_Y = m * const_diagonal
#         # else:
#         diag_X = torch.diag(K_XX)                       # (m,)
#         diag_Y = torch.diag(K_YY)                       # (m,)
#         sum_diag_X = torch.sum(diag_X)
#         sum_diag_Y = torch.sum(diag_Y)

#         Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
#         Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
#         K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

#         Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
#         Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
#         K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

#         #if biased:
#         mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)+ (Kt_YY_sum + sum_diag_Y) / (m * m)- 2.0 * K_XY_sum / (m * m))
#         # else:
#         #     mmd2 = (Kt_XX_sum / (m * (m - 1))
#         #         + Kt_YY_sum / (m * (m - 1))
#         #         - 2.0 * K_XY_sum / (m * m))

#         return mmd2
        

# # def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
# #     # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
# #     return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


# class mix_rbf_mmd2_joint_1_feature_1_label(torch.nn.Module):
#     def __init__(self,sigma_list_effect):
#         super(mix_rbf_mmd2_joint_1_feature_1_label, self).__init__()
#         self.rbf_effect=mix_rbf_kernel_class(sigma_list=sigma_list_effect)
#         self.kern_linear=_mix_linear_kernel
        
        
#     def forward(self,X_effect_hat,X_effect_true,Y_lab_hat,Y_lab_true):
        
        
#         K_XX, K_XY, K_YY, d = self.rbf_effect(X_effect_hat,X_effect_true)
#         K_XX1, K_XY1, K_YY1 = self.kern_linear(Y_lab_hat,Y_lab_true)
#         K_XX = K_XX * K_XX1
#         K_YY = K_YY * K_YY1
#         K_XY = K_XY * K_XY1
#         m = K_XX.size(0)    # assume X, Y are same shape

#         # # Get the various sums of kernels that we'll use
#         # # Kts drop the diagonal, but we don't need to compute them explicitly
#         # if const_diagonal is not False:
#         #     diag_X = diag_Y = const_diagonal
#         #     sum_diag_X = sum_diag_Y = m * const_diagonal
#         # else:
#         diag_X = torch.diag(K_XX)                       # (m,)
#         diag_Y = torch.diag(K_YY)                       # (m,)
#         sum_diag_X = torch.sum(diag_X)
#         sum_diag_Y = torch.sum(diag_Y)

#         Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
#         Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
#         K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

#         Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
#         Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
#         K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

#         #if biased:
#         mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)+ (Kt_YY_sum + sum_diag_Y) / (m * m)- 2.0 * K_XY_sum / (m * m))
#         # else:
#         #     mmd2 = (Kt_XX_sum / (m * (m - 1))
#         #         + Kt_YY_sum / (m * (m - 1))
#         #         - 2.0 * K_XY_sum / (m * m))

#         return mmd2
    
    

    
# class mix_rbf_mmd2_joint_regress_2_feature(torch.nn.Module):
#     def __init__(self,sigma_list_effect,sigma_list_cause):
#         super(mix_rbf_mmd2_joint_regress_2_feature, self).__init__()
#         self.rbf_effect=mix_rbf_kernel_class(sigma_list=sigma_list_effect)
#         self.rbf_cause=mix_rbf_kernel_class(sigma_list=sigma_list_cause)
        

#     def forward(self,X_e_hat,X_e_true,X_c_hat,X_c_true):
#         K_XX, K_XY, K_YY, d = self.rbf_effect(X_e_hat,X_e_true)
#         K_XX1, K_XY1, K_YY1, d1 = self.rbf_cause(X_c_hat,X_c_true)
#         K_XX = K_XX * K_XX1
#         K_YY = K_YY * K_YY1
#         K_XY = K_XY * K_XY1

#         m = K_XX.size(0)    # assume X, Y are same shape

#         # else:
#         diag_X = torch.diag(K_XX)                       # (m,)
#         diag_Y = torch.diag(K_YY)                       # (m,)
#         sum_diag_X = torch.sum(diag_X)
#         sum_diag_Y = torch.sum(diag_Y)

#         Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
#         Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
#         K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

#         Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
#         Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
#         K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

#         #if biased:
#         mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)+ (Kt_YY_sum + sum_diag_Y) / (m * m)- 2.0 * K_XY_sum / (m * m))
#         # else:
#         #     mmd2 = (Kt_XX_sum / (m * (m - 1))
#         #         + Kt_YY_sum / (m * (m - 1))
#         #         - 2.0 * K_XY_sum / (m * m))

#         return mmd2






#--------------------JIT


#@ remove dependence on sigma list, so can ve pass in as tensor direct

#custom compiled kern for if we need to do many mmd per run, speedup

def _mix_linear_kernel(X, Y):
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    K = torch.mm(Z, Z.t())

    return K[:m, :m], K[:m, m:], K[m:, m:]



class mix_rbf_kernel_class(torch.nn.Module):
    
    def __init__(self):
        super(mix_rbf_kernel_class, self).__init__()
        #self.sigma_list=sigma_list
        

    def forward(self,X,Y,sigma_list):
        assert (X.size(0) == Y.size(0))
        m = X.size(0)
        Z = torch.cat((X, Y), 0)
        ZZT = torch.mm(Z, Z.t())
        diag_ZZT = torch.diag(ZZT).unsqueeze(1)
        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()
        K=torch.sum(torch.exp(exponent[None,:,:].expand(sigma_list.shape[0],-1,-1) * -(1.0 / (2 * sigma_list ** 2))[:,None,None]), dim=0)
        
        return K[:m, :m], K[:m, m:], K[m:, m:], sigma_list.shape[0]



class mmd2_class(torch.nn.Module):
    
    def __init__(self):
        super(mmd2_class, self).__init__()
        #self.rbf_kernel=mix_rbf_kernel_class()
        
    def forward(self,K_XX, K_XY, K_YY):
        #K_XX, K_XY, K_YY, d = self.rbf_kernel(X, Y,sigma_list)
        m = K_XX.size(0)    # assume X, Y are same shape

        # # Get the various sums of kernels that we'll use
        # # Kts drop the diagonal, but we don't need to compute them explicitly
        # if const_diagonal is not False:
        #     diag_X = diag_Y = const_diagonal
        #     sum_diag_X = sum_diag_Y = m * const_diagonal
        # else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

        Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
        Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
        K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

        Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
        Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
        K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

        #if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)+ (Kt_YY_sum + sum_diag_Y) / (m * m)- 2.0 * K_XY_sum / (m * m))
        # else:
        #     mmd2 = (Kt_XX_sum / (m * (m - 1))
        #         + Kt_YY_sum / (m * (m - 1))
        #         - 2.0 * K_XY_sum / (m * m))

        return mmd2
        



class mix_rbf_mmd2_class(torch.nn.Module):
    
    def __init__(self):
        super(mix_rbf_mmd2_class, self).__init__()
        self.rbf_kernel=mix_rbf_kernel_class()
        
    def forward(self,X,Y,sigma_list):
        K_XX, K_XY, K_YY, d = self.rbf_kernel(X, Y,sigma_list)
        m = K_XX.size(0)    # assume X, Y are same shape

        # # Get the various sums of kernels that we'll use
        # # Kts drop the diagonal, but we don't need to compute them explicitly
        # if const_diagonal is not False:
        #     diag_X = diag_Y = const_diagonal
        #     sum_diag_X = sum_diag_Y = m * const_diagonal
        # else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

        Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
        Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
        K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

        Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
        Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
        K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

        #if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)+ (Kt_YY_sum + sum_diag_Y) / (m * m)- 2.0 * K_XY_sum / (m * m))
        # else:
        #     mmd2 = (Kt_XX_sum / (m * (m - 1))
        #         + Kt_YY_sum / (m * (m - 1))
        #         - 2.0 * K_XY_sum / (m * m))

        return mmd2
        

# def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
#     # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
#     return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


class mix_rbf_mmd2_joint_1_feature_1_label(torch.nn.Module):
    def __init__(self):
        super(mix_rbf_mmd2_joint_1_feature_1_label, self).__init__()
        self.rbf_effect=mix_rbf_kernel_class()
        self.kern_linear=_mix_linear_kernel
        
        
    def forward(self,X_effect_hat,X_effect_true,Y_lab_hat,Y_lab_true,sigma_list_effect):
        
        
        K_XX, K_XY, K_YY, d = self.rbf_effect(X_effect_hat,X_effect_true,sigma_list_effect)
        K_XX1, K_XY1, K_YY1 = self.kern_linear(Y_lab_hat,Y_lab_true)
        K_XX = K_XX * K_XX1
        K_YY = K_YY * K_YY1
        K_XY = K_XY * K_XY1
        m = K_XX.size(0)    # assume X, Y are same shape

        # # Get the various sums of kernels that we'll use
        # # Kts drop the diagonal, but we don't need to compute them explicitly
        # if const_diagonal is not False:
        #     diag_X = diag_Y = const_diagonal
        #     sum_diag_X = sum_diag_Y = m * const_diagonal
        # else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

        Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
        Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
        K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

        Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
        Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
        K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

        #if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)+ (Kt_YY_sum + sum_diag_Y) / (m * m)- 2.0 * K_XY_sum / (m * m))
        # else:
        #     mmd2 = (Kt_XX_sum / (m * (m - 1))
        #         + Kt_YY_sum / (m * (m - 1))
        #         - 2.0 * K_XY_sum / (m * m))

        return mmd2
    
    

    
class mix_rbf_mmd2_joint_regress_2_feature(torch.nn.Module):
    def __init__(self):
        super(mix_rbf_mmd2_joint_regress_2_feature, self).__init__()
        self.rbf_effect=mix_rbf_kernel_class()
        self.rbf_cause=mix_rbf_kernel_class()
        

    def forward(self,X_e_hat,X_e_true,X_c_hat,X_c_true,sigma_list_effect,sigma_list_cause):
        K_XX, K_XY, K_YY, d = self.rbf_effect(X_e_hat,X_e_true,sigma_list_effect)
        K_XX1, K_XY1, K_YY1, d1 = self.rbf_cause(X_c_hat,X_c_true,sigma_list_cause)
        K_XX = K_XX * K_XX1
        K_YY = K_YY * K_YY1
        K_XY = K_XY * K_XY1

        m = K_XX.size(0)    # assume X, Y are same shape

        # else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

        Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
        Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
        K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

        Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
        Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
        K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

        #if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)+ (Kt_YY_sum + sum_diag_Y) / (m * m)- 2.0 * K_XY_sum / (m * m))
        # else:
        #     mmd2 = (Kt_XX_sum / (m * (m - 1))
        #         + Kt_YY_sum / (m * (m - 1))
        #         - 2.0 * K_XY_sum / (m * m))

        return mmd2




    
class mix_rbf_mmd2_joint_regress_2_feature_1_label(torch.nn.Module):
    def __init__(self):
        super(mix_rbf_mmd2_joint_regress_2_feature_1_label, self).__init__()
        self.rbf_effect=mix_rbf_kernel_class()
        self.rbf_cause=mix_rbf_kernel_class()
        self.kern_linear=_mix_linear_kernel

        

    def forward(self,X_e_hat,X_e_true,X_c_hat,X_c_true,Y_hat,Y_true,sigma_list_effect,sigma_list_cause):
        K_XX, K_XY, K_YY, d = self.rbf_effect(X_e_hat,X_e_true,sigma_list_effect)
        K_XX1, K_XY1, K_YY1, d1 = self.rbf_cause(X_c_hat,X_c_true,sigma_list_cause)
        K_XX2, K_XY2, K_YY2 = self.kern_linear(Y_hat, Y_true)
        
        
        K_XX = K_XX * K_XX1 * K_XX2
        K_YY = K_YY * K_YY1 * K_YY2
        K_XY = K_XY * K_XY1 * K_XY2
        
        #K_XX = K_XX * K_XX1
        #K_YY = K_YY * K_YY1
        #K_XY = K_XY * K_XY1

        m = K_XX.size(0)    # assume X, Y are same shape

        # else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

        Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
        Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
        K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

        Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
        Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
        K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

        #if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)+ (Kt_YY_sum + sum_diag_Y) / (m * m)- 2.0 * K_XY_sum / (m * m))
        # else:
        #     mmd2 = (Kt_XX_sum / (m * (m - 1))
        #         + Kt_YY_sum / (m * (m - 1))
        #         - 2.0 * K_XY_sum / (m * m))

        return mmd2






class mix_rbf_mmd2_joint_regress_3_feature_1_label(torch.nn.Module):
    def __init__(self):
        super(mix_rbf_mmd2_joint_regress_3_feature_1_label, self).__init__()
        self.rbf=mix_rbf_kernel_class()
        #self.rbf_cause=mix_rbf_kernel_class()
        self.kern_linear=_mix_linear_kernel

        

    def forward(self,X_e_hat,X_e_true,X_c_hat,X_c_true,X_s_hat,X_s_true,Y_hat,Y_true,sigma_list_effect,sigma_list_cause,sigma_list_spouse):
        K_XX, K_XY, K_YY, d = self.rbf(X_e_hat,X_e_true,sigma_list_effect)
        K_XX1, K_XY1, K_YY1, d1 = self.rbf(X_c_hat,X_c_true,sigma_list_cause)
        K_XX3, K_XY3, K_YY3, d3 = self.rbf(X_s_hat,X_s_true,sigma_list_spouse)
        K_XX2, K_XY2, K_YY2 = self.kern_linear(Y_hat, Y_true)
        
        
        K_XX = K_XX * K_XX1 * K_XX2 * KXX3
        K_YY = K_YY * K_YY1 * K_YY2 * KYY3
        K_XY = K_XY * K_XY1 * K_XY2 * KXY3
        
        #K_XX = K_XX * K_XX1
        #K_YY = K_YY * K_YY1
        #K_XY = K_XY * K_XY1

        m = K_XX.size(0)    # assume X, Y are same shape

        # else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

        Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
        Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
        K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

        Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
        Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
        K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

        #if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)+ (Kt_YY_sum + sum_diag_Y) / (m * m)- 2.0 * K_XY_sum / (m * m))
        # else:
        #     mmd2 = (Kt_XX_sum / (m * (m - 1))
        #         + Kt_YY_sum / (m * (m - 1))
        #         - 2.0 * K_XY_sum / (m * m))

        return mmd2








# lab_loss = self.dop['mix_rbf_mmd2_joint_regress_2_feature'](x_hat,
#                                             target_x,
#                                             conditional_x,
#                                             conditional_x,
#                                             y,
#                                             y,
#                                             self.sigma_list_target_x,
#                                             self.sigma_list_cond_x)