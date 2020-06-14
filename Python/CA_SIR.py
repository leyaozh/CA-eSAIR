import os
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from urllib.request import urlopen
import json
from geopy.distance import distance
import plotly as pl
import math
import seaborn as sns
import matplotlib.pyplot as plt

########################################################################################################################
########################################################################################################################
class CA_SIR:
    def __init__(self, epsilon, nu, a, eta, mMat, pMat0, pMat, Sp, Ip, Rp, Inum, pred, thres):
        self.epsilon = epsilon  ##transmission rate
        self.nu = nu  ##removal rate
        self.a = a  ##connection factor
        self.eta = eta  ##parameter to tune omega (prediction accuracy)
        self.mMat = mMat  ##movement factor
        self.pMat0 = pMat0  ##1/population_c
        self.pMat = pMat  ##population_c'/population_c
        self.Sp = Sp  ##a vector containing I at t-1 for all the counties
        self.Ip = Ip  ##a vector containing I at t-1 for all the counties
        self.Rp = Rp  ##a vector containing I at t-1 for all the counties
        self.Inum = Inum  ##observed number of infected cases on t0
        self.pred = pred  ##character: one-day or t_day ahead risk prediction
        self.thres = thres

    def weightSum(self):

        wetVec = np.zeros(len(self.Ip))
        if self.pred == "one":
            for i in range(len(self.Ip)):
                m = self.mMat[i, :]
                nni = self.pMat0[i, :] * self.Inum
                ##assume a threshold of probability for infected individuals meeting individuals from another county
                wet = np.where(nni > self.thres, self.thres, nni)
                wetVec[i] = np.sum(self.a * self.epsilon * self.eta * m * wet)
        elif self.pred == "t":
            for i in range(len(self.Ip)):
                m = self.mMat[i, :]
                nni = self.pMat[i, :]
                wet = np.where(nni > self.thres, self.thres, nni)
                wetVec[i] = np.sum(self.a * self.epsilon * self.eta * m * wet * self.Ip)

        self.wetVec = wetVec

    def update(self):

        St = self.Sp-self.epsilon*self.Sp*self.Ip-self.Sp*self.wetVec
        It = (1-self.nu)*self.Ip+self.epsilon*self.Sp*self.Ip+self.Sp*self.wetVec
        Rt = self.Rp+self.nu*self.Ip

        self.St = St
        self.It = It
        self.Rt = Rt


    
########################################################################################################################
class CA_SAIR:
    def __init__(self, epsilon, nu, alpha, pi, eta, dMat, iMat, pMat0, pMat, Sp, Ap, Ip, Rp, Inum, pred, thres):
        self.epsilon = epsilon  ##transmission rate
        self.nu = nu  ##removal rate
        self.alpha = alpha  ##used to adjust the prevalence of people with antibody (this is calculated outside)
        self.pi = pi  ##transmission modifier (this is calculated outside)
        self.eta = eta  ##parameter to tune omega (prediction accuracy)
        self.dMat = dMat ##distance matrix
        self.iMat = iMat  ##inter-county mobility factor matrix
        self.pMat0 = pMat0  ##1/population_c
        self.pMat = pMat  ##population_c'/population_c
        self.Sp = Sp  ##a vector containing S at t-1 for all the counties
        self.Ap = Ap  ##a vector containing A at t-1 for all the counties
        self.Ip = Ip  ##a vector containing I at t-1 for all the counties
        self.Rp = Rp  ##a vector containing R at t-1 for all the counties
        self.Inum = Inum  ##observed number of infected cases on t0
        self.pred = pred  ##character: one-day or t_day ahead risk prediction
        self.thres = thres

    def weightSum(self):

        omega = np.zeros((len(self.Ip), len(self.Ip)))
        omega = self.iMat * self.dMat ** self.eta  
        self.omega = omega
        
        wetVec = np.zeros(len(self.Ip))
        if self.pred == "one":
            for i in range(len(self.Ip)):
                nni = self.pMat0[i, :] * self.Inum
                ##assume a threshold of probability for infected individuals meeting individuals from another county
                wet = np.where(nni > self.thres, self.thres, nni)
                wetVec[i] = np.sum(omega[i, :] * self.epsilon * wet * self.Ip)
        elif self.pred == "t":
            for i in range(len(self.Ip)):
                nni = self.pMat[i, :]
                wet = np.where(nni > self.thres, self.thres, nni)
                wetVec[i] = np.sum(omega[i, :] * self.epsilon * wet * self.Ip)

        self.wetVec = wetVec

    def update(self):

        At = self.Ap+self.alpha*self.Sp
        St = (1-self.alpha)*self.Sp-self.epsilon*self.pi*self.Sp*self.Ip-self.pi*self.Sp*self.wetVec
        It = (1-self.nu)*self.Ip+self.epsilon*self.pi*self.Sp*self.Ip+self.pi*self.Sp*self.wetVec
        Rt = self.Rp+self.nu*self.Ip

        self.At = At
        self.St = St
        self.It = It
        self.Rt = Rt

########################################################################################################################
def goodness_fit_test(obsI, predI, pop):
    wt = pop/sum(pop)
    #PE = sum((obsI-predI) ** 2 * wt)
    PE = sum(abs(obsI-predI) * wt)
    return(PE)

########################################################################################################################
def riskFunc(Imat):

    riskCum = Imat[:,1]
    for i in range(2, Imat.shape[1]):
        rr = 1
        for j in range(1,i):
            rr = rr*(1-Imat[:,j])

        riskCum = riskCum+rr*Imat[:,i]

    return(riskCum)

########################################################################################################################
def riskTravel(Imat, t0, c_travel, fips):
    
    time_travel = len(c_travel)
    Itravel = np.full(time_travel+1, 1.0)
    index0 = np.full(time_travel+1, 0)
    riskCum = np.full(Imat.shape[0], 0.0)

    for t in range(1, time_travel+1):
        index0[t] = fips.index(c_travel[t-1])
        Itravel[t] = Imat[index0[t], t0+t]
        if t==1:
            riskCum[index0[t]] = Itravel[t]
        else:
            rt = 1
            for i in range(1,t):
                rt = rt * (1- Itravel[i])
            riskCum[index0[t]] = riskCum[index0[t-1]] + rt*Itravel[t]
            
    return(riskCum)

########################################################################################################################
def plot_risk(fips, risk, scope, plotName):

    values = [round(10000 * risk[elem], 2) for elem in range(len(risk))]
    cat = [np.quantile(values, 0.2), np.quantile(values, 0.4), np.quantile(values, 0.6), np.quantile(values, 0.8)]
    colorscale = ["#d9e3ec", "#C5D6E9", "#B9C0DD", "#B79ACA", "#B472B1"]

    if scope == "usa":
        fig = ff.create_choropleth(fips=fips, values=values, legend_title='Infection Risk (1/10,000)',
                                   binning_endpoints=cat, colorscale=colorscale, plot_bgcolor='#f1f1f2')
    else:
        scope = [scope]
        fig = ff.create_choropleth(fips=fips, values=values, scope=scope, legend_title='Infection Risk (1/10,000)',
                                   binning_endpoints=cat, colorscale=colorscale, plot_bgcolor='#f1f1f2')

    fig.layout.template = None
    fig.show()
    pl.offline.plot(fig, filename=plotName+".html")
    fig.write_image("fig.png", width=2048, height=1024, scale=2)

########################################################################################################################
def plot_travel(data, Itravel, scope, plotName):

    index = []
    for i in range(len(scope)):
        index = index + data.index[data['state'] == scope[i]].tolist()

    fips = data['fips'][index]
    values = [round(10000 * Itravel[elem], 2) for elem in index]
    fig = ff.create_choropleth(fips=fips, values=values, scope=scope, legend_title='Infection Risk (1/10,000)',
                               colorscale=['#cfd0d2',"#cf90a2",'#8f5483', '#744475'], plot_bgcolor='#f1f1f2')

    fig.layout.template = None
    fig.show()
    pl.offline.plot(fig, filename=plotName+".html")
    fig.write_image("fig.png", width=2048, height=1024, scale=2)





