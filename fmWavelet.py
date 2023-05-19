# -*- coding:utf-8 -*-

import linecache
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class FMwavelet:
    def __init__(self):
        # get file name
        current_working_dir = os.getcwd()
        feff_name = str(f"{current_working_dir}\\path.txt")
        para_name = str(f"{current_working_dir}\\parameters.txt")
        # load parameters
        readPara(para_name)
        path_data = []
        # get r_eff calculated by feff in the feffxxxx.dat file
        # get k_mesh, scattering amptitude, phase, lamda and reduce factor in feffxxxx.dat file
        if Parameters.feff_ver == 6 or Parameters.feff_ver == 8:
            count = 1
            with open(feff_name, encoding="utf-8") as file:
                for line in file:
                    if re.search('----------', line):
                        break
                    else:
                        count += 1
            Parameters.n_deg = float(linecache.getline(feff_name, count + 1).split()[1])
            Parameters.r_eff = float(linecache.getline(feff_name, count + 1).split()[2])
            path_data = np.loadtxt(feff_name, skiprows=count + 5)
        else:
            print("wrong")
            quit()
        interp_path_data = interpValueOfFeff(path_data, Parameters.dk_data, Parameters.kcenter_shift)
        amp_value_data = amp1(interp_path_data, Parameters.k_weighted, Parameters.dwf, Parameters.r_eff,
                              Parameters.e_lambda)
        amp_gau_value_data = insertGauEnvelope(amp_value_data, Parameters.kmin_feff, Parameters.kmax_feff,
                                               Parameters.gaussian_sigma)
        showValue(amp_gau_value_data, 6)
        fm_wavelet = pathWavelet(amp_gau_value_data, Parameters.r_eff, Parameters.smin, Parameters.smax, Parameters.ds,
                                 Parameters.kmin_data, Parameters.kmax_data, Parameters.dk_data, Parameters.n_deg)


class Parameters:
    feff_ver = 6  # only 6 or 8
    k_weighted = 2  # only 1/2/3
    kmin_feff = 3  # kmin of feff path
    kmax_feff = 9  # kmax of feff path
    kcenter_shift = 0.0  # k shift of feff path
    gaussian_sigma = 0.5  # gaussian envelope (Å-1)
    kmin_data = 0  # kmin of EXAFS data
    kmax_data = 10.00  # kmax of EXAFS data
    dk_data = 0.05  # step size of k in dataset
    smin = 0.80  # scale min you want to scan
    smax = 1.20  # scale max you want to scan
    ds = 0.02  # step size of scale
    dwf = 0.000  # Debye–Waller factor
    e_lambda = 0  # introduce mean free path of electron when e_lambda=1, else e_lambda=0
    r_eff = 3  # r_eff in feff file
    n_deg = 1  # r_eff in feff file


def readPara(para_name):
    # read parameters
    para_load = np.loadtxt(para_name, usecols=1)
    Parameters.feff_ver = para_load[0]
    Parameters.k_weighted = para_load[1]
    Parameters.kmin_feff = para_load[2]
    Parameters.kmax_feff = para_load[3]
    Parameters.kcenter_shift = para_load[4]
    Parameters.gaussian_sigma = para_load[5]
    Parameters.kmin_data = para_load[6]
    Parameters.kmax_data = para_load[7]
    Parameters.dk_data = para_load[8]
    Parameters.smin = para_load[9]
    Parameters.smax = para_load[10]
    Parameters.ds = para_load[11]
    Parameters.dwf = para_load[12]
    Parameters.e_lambda = para_load[13]


def valueList(v_min, v_max, v_delta):
    value_list = np.arange(v_min, v_max + v_delta, v_delta)
    return value_list


def interpValueOfFeff(feff_value, dk, k_shift):
    col_limit = feff_value.shape[1]  # get column num
    # introduce k shift
    k_feff = feff_value[:, 0]
    k_shift = k_shift * np.ones_like(k_feff)
    k_feff = np.add(k_feff, k_shift)
    # interpolate k value
    k_mesh = valueList(k_feff[0], k_feff[-1], dk)  # create k value with deltak=0.05
    final_data = k_mesh
    # interpolate value along column and finally combine
    value_interp = []
    for i in range(1, col_limit):
        f = interpolate.interp1d(k_feff, feff_value[:, i], kind=3, fill_value='extrapolate')
        value_temp = f(k_mesh)
        value_interp.append(value_temp)
    for i in range(len(value_interp)):
        final_data = np.column_stack((final_data, value_interp[i]))
    # save final data
    current_working_dir = os.getcwd()
    np.savetxt(str(f"{current_working_dir}\\interpValue.txt"), final_data, fmt='%f', delimiter=" ")
    return final_data


def amp1(interp_value, kw, dwf, r_eff, e_lambda: int):
    # create kw value (containing sphere wave condition, which means the kw should larger than 1)
    k_data = interp_value[:, 0]
    kw_data = 1 / (r_eff * r_eff) * np.ones_like(k_data)
    if kw == 0:
        print("wrong")
        quit()
    elif kw == 1:
        kw_data = kw_data
    else:
        while kw > 1:
            kw_data = np.multiply(kw_data, k_data)
            kw -= 1
    # calculate backscattering amptitude / phase according to Matthew Newville
    f_k = np.multiply(interp_value[:, 2], interp_value[:, 4])
    phase_k = np.add(interp_value[:, 1], interp_value[:, 3])
    # calculate reduction from mean free path of electron according to Matthew Newville
    lambda_value = interp_value[:, 5]
    if e_lambda == 0:
        lambda_reduction = np.ones_like(k_data)
    else:
        lambda_reduction = np.array([np.exp(-2 * r_eff / la) for la in lambda_value])
    # calculate reduction from Debye-Waller factor
    dwf_reducton = np.array([np.exp(-2 * k * k * dwf) for k in k_data])
    # combine
    amp_value = np.column_stack((k_data, phase_k, kw_data, f_k, lambda_reduction, dwf_reducton))
    # get combine column num
    col_limit = amp_value.shape[1]  # get column num
    # multiply all amp factor together
    amp_k = np.ones_like(k_data)
    for i in range(2, col_limit):
        amp_k = np.multiply(amp_k, amp_value[:, i])
    # multiply oscillation(imag part sin) with amp
    wave_show = np.array([amp * np.sin(2 * r_eff * k + ph) for amp, k, ph in zip(amp_k, k_data, phase_k)])
    # combine
    amp_with_wave_show = np.column_stack((amp_value, amp_k, wave_show))
    # save data
    current_working_dir = os.getcwd()
    np.savetxt(str(f"{current_working_dir}\\ampValue.txt"), amp_with_wave_show, fmt='%f', delimiter=" ")
    return amp_with_wave_show


def insertGauEnvelope(amp1_data, point_min, point_max, sigma):
    def thresholdOfGau(value):
        if value < 1E-5:
            value = 0
        else:
            value = value
        return value
    # constant
    # c = 1 / (np.sqrt(2 * np.pi) * sigma)
    c = 1
    # build up gaussian envelope with sigma=sigma
    gau_envelope = []
    adp = gau_envelope.append
    k = amp1_data[:, 0]
    for i in range(len(k)):
        k_value = k[i]
        if k_value < point_min:
            temp = c * np.exp(-1 * (k_value - point_min) * (k_value - point_min) / (2 * sigma * sigma))
            temp = thresholdOfGau(temp)
            adp(temp)
        elif k_value > point_max:
            temp = c * np.exp(-1 * (k_value - point_max) * (k_value - point_max) / (2 * sigma * sigma))
            temp = thresholdOfGau(temp)
            adp(temp)
        else:
            adp(c)
    # multiply Gau envelope with amp
    amp_gau = np.multiply(amp1_data[:, 6], gau_envelope)
    # combine
    amp_with_gau = np.column_stack((amp1_data, amp_gau))
    # save data
    current_working_dir = os.getcwd()
    np.savetxt(str(f"{current_working_dir}\\ampWithGau.txt"), amp_with_gau, fmt='%f', delimiter=" ")
    return amp_with_gau


def pathWavelet(amp_gau_value, r_eff, smin, smax, ds, kmin, kmax, dk, n_deg):
    i_num = complex(0, 1)
    k_data = amp_gau_value[:, 0]
    # preparation of adding 0 for analysis area
    k_interp = valueList(kmin - 10, kmax + 10, dk)
    phase_data = amp_gau_value[:, 1]
    amp_data = n_deg * amp_gau_value[:, 8]
    s_value = valueList(smin, smax, ds)
    # create oscillation with phase shift along scale
    feff_morlet_wavelet = np.array([np.interp(k_interp, k_data, [amp * np.exp(i_num * s * (2 * r_eff * k + ph))
                                                                 for amp, k, ph in zip(amp_data, k_data, phase_data)],
                                              left=0.0, right=0.0)
                                    for s in s_value])
    # create energy coef according to Harald Funke
    energy_coef = np.array([np.sqrt(s) for s in s_value])
    # combine data
    final_data = k_interp
    final_data_plt = k_interp
    for i in range(len(feff_morlet_wavelet)):
        final_data = np.column_stack((final_data, feff_morlet_wavelet[i]))
        final_data_plt = np.column_stack((final_data_plt, feff_morlet_wavelet[i].imag))
    # save data
    current_working_dir = os.getcwd()
    np.savetxt(str(f"{current_working_dir}\\temp_MotherWavelet.txt"), final_data, fmt='%f', delimiter=" ")
    np.savetxt(str(f"{current_working_dir}\\energy_coef.txt"), energy_coef, fmt='%f', delimiter=" ")
    return True


def showValue(data_value, start_col):
    plt.close(1)
    plt.grid()
    plt.xlim((data_value[:, 0][0], data_value[:, 0][-1]))
    plt.xlabel("k (Å$^{-1}$)")
    plt.ylabel("value")
    k = data_value[:, 0]
    col_limits = data_value.shape[1]
    for i in range(start_col, col_limits):
        temp_value = data_value[:, i]
        plt.plot(k, temp_value, linewidth=1.2, label=i+1)
    plt.legend(loc='upper right', title="Column")
    plt.show()


if __name__ == '__main__':
    FMwavelet()
