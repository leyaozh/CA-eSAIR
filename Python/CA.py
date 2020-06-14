import os
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from urllib.request import urlopen
import json
from geopy.distance import distance
#from CA_SIR import CA_SIR
import CA_SIR as CA_SIR
import plotly as pl
import math
import seaborn as sns
import matplotlib.pyplot as plt

########################################################################################################################
##os.chdir("") ##change working directory

########################################################################################################################
## read in the fips centroid data & the county population data
fipsData = pd.read_csv("fips_centroid_5_2.csv")
popData = pd.read_csv("population.csv")

fipsData['fips'] = fipsData['fips'].apply(str)
popData['fips'] = popData['fips'].apply(str)

## make all the fips code to 5-letter string
fipsData['fips'] = np.where(fipsData['fips'].str.len() == 4, '0'+fipsData['fips'], fipsData['fips'])
popData['fips'] = np.where(popData['fips'].str.len() == 4, '0'+popData['fips'], popData['fips'])

## merge fipsData with popData
data = pd.merge(fipsData, popData, on='fips')

## observed values for calculating SSPE
fipsData_obs = pd.read_csv("fips_centroid_5_3.csv")
fipsData_obs['fips'] = fipsData_obs['fips'].apply(str)
fipsData_obs['fips'] = np.where(fipsData_obs['fips'].str.len() == 4, '0'+fipsData_obs['fips'], fipsData_obs['fips'])
data_obs = pd.merge(fipsData_obs, popData, on='fips')

########################################################################################################################
## calculate the geodesic distance between two counties
'''
distData = np.zeros((data.shape[0], data.shape[0]))

for i in range(data.shape[0]):
    for j in range(i,data.shape[0]):
        d = (distance(data.iloc[i,3:4], data.iloc[j,3:4]).m)/1000  ##km (unit)
        distData[i, j] = d

        if i != j:
            distData[j, i] = d
'''


## read in distance data (= distance + airport size factor * (airport distance))
distData = pd.read_csv("distance_final.csv", header = None)
distData = distData.to_numpy()


## read in inter-county mobility factor 
interData = pd.read_csv("interfactor.csv", header = None)
interData = interData.to_numpy()

            
## read in pi value
pi = pd.read_csv("pi_value.csv", header = None)
pi = pi.to_numpy()

## read in alpha value
## alpha_fips = pd.read_csv("alpha_fips.csv")
## alpha = alpha_fips['alpha']
alpha = pd.read_csv("alpha.csv")
alpha = alpha.to_numpy()


########################################################################################################################
##set parameters
t = 7  ##7-day ahead prediction
time = t+1  ##dimension of saved prediction results

phi = np.full(t, 0.005)  ##*****can be further discussed

epsilon = 0.077  ##estimated transmission rate from eSIR model
nu = 0.023  ##estimated removal rate from eSIR model

eta_list = [0.1,10,20,30,40,50,60,70,80, 90, 100]
eta_list = [0.1 * i for i in list(range(1,101))]

pop = data['count']


dMat = np.vstack([np.where(distData[i,:] == 0, 0, np.exp(-distData[i,:])) for i in range(data.shape[0])])  ##distance parameter

pMat0 = np.vstack([1/pop[i] for i in range(data.shape[0])])
pMat = np.vstack([pop/pop[i] for i in range(data.shape[0])])


## state with eSIR estimation
state39 = ["AZ", "AR", "CA", "CO", "CT", "DE", "DC", "GA", "IL", "IN", "IA", "KS", "KY",
           "ME", "MD", "MA", "MI", "MN", "MO", "MT", "NV", "NH", "NJ", "NM", "NY", "NC", 
           "ND", "OH", "OK", "PA", "SD", "TN", "TX", "UT", "VA", "WA", "WV", "WI", "WY"]
temp = set(state39)

index39 = [i for i, val in enumerate(data['state']) if val in temp] 

########################################################################################################################
##matrix to save results
Smat = np.full((data.shape[0], time, len(eta_list)), np.nan)
Imat = np.full((data.shape[0], time, len(eta_list)), np.nan)
Rmat = np.full((data.shape[0], time, len(eta_list)), np.nan)
Amat = np.full((data.shape[0], time, len(eta_list)), np.nan)

I0 = np.full(data.shape[0],1)
Inum = data['confirmed']
I0 = Inum/pop

##read Smat, Imat, Rmat, Amat[:,0,:] data

ASIR0 = pd.read_csv("ASIR0_fips.csv")
ASIR0_upper = pd.read_csv("ASIR0_upper.csv")
ASIR0_lower = pd.read_csv("ASIR0_lower.csv")

for al in range(len(eta_list)):
    Imat[:, 0 ,al] = ASIR0['I0']
    Amat[:, 0 ,al] = ASIR0['A0']
    Rmat[:, 0 ,al] = ASIR0['R0']
    for i in range(data.shape[0]):
        if pop[i] > 100000:
            Imat[i,0,al] = I0[i]
        elif pop[i] > 50000:
            Imat[i,0,al] = (I0[i] + Imat[i, 0, al])/2
        elif pop[i] > 25000:
            Imat[i,0,al] = I0[i]/4.0 + Imat[i, 0, al]*3.0/4
    Smat[:, 0 ,al] = 1 - Imat[:, 0 ,al] - Amat[:, 0 ,al] - Rmat[:, 0 ,al]
    
########################################################################################################################
## one-day risk prediction
PE_one_list_num = np.full(len(eta_list), np.nan) ## SSPE for number of infected people
PE_one_list_rate = np.full(len(eta_list), np.nan) ## SSPE for infected rate

for al in range(len(eta_list)):
    caModel0 = CA_SIR.CA_SAIR(epsilon, nu, alpha[:,0], pi[:,0], eta_list[al], dMat, interData, pMat0, pMat, 
                              Smat[:,0,al], Amat[:,0,al], Imat[:,0,al], Rmat[:,0,al],
                              Inum, pred="one", thres=1e-3)
    caModel0.weightSum()
    caModel0.update()
    A1, S1, I1, R1 = caModel0.At, caModel0.St, caModel0.It, caModel0.Rt

    Amat[:,1,al] = A1
    Smat[:,1,al] = S1
    Imat[:,1,al] = I1
    Rmat[:,1,al] = R1

    PE_one_list_num[al] = CA_SIR.goodness_fit_test(data_obs['confirmed'][index39],I1[index39] * pop[index39], pop[index39]) 
    PE_one_list_rate[al] = CA_SIR.goodness_fit_test(data_obs['confirmed'][index39]/pop[index39] ,I1[index39], pop[index39])

index_one = np.where(PE_one_list_rate == np.amin(PE_one_list_rate))[0]

np.savetxt("S1.csv", Smat[:,1,index_one], delimiter=",")
np.savetxt("I1.csv", Imat[:,1,index_one], delimiter=",")
np.savetxt("R1.csv", Rmat[:,1,index_one], delimiter=",")

np.savetxt("index.csv", index39, delimiter = ",")
np.savetxt("PE.csv", PE_one_list_rate, delimiter = ",")


########################################################################################################################
## t-day risk prediction

al = 0
for t in range(1, time):
    if t == 1:
        caModel = CA_SIR.CA_SAIR(epsilon, nu, alpha[:,t-1], pi[:, t-1], 35, dMat, interData, pMat0, pMat, Smat[:,t-1,al], Amat[:,t-1,al], Imat[:,t-1,al],
                                 Rmat[:,t-1,al], Inum, pred="one", thres=1e-3)
        caModel.weightSum()
        caModel.update()
        Amat[:,t,al], Smat[:,t,al], Imat[:,t,al], Rmat[:,t,al] = caModel.At, caModel.St, caModel.It, caModel.Rt
    else:
        caModel = CA_SIR.CA_SAIR(epsilon, nu, alpha[:,t-1], pi[:, t-1], 35, dMat, interData, pMat0, pMat, Smat[:,t-1,al], Amat[:,t-1,al], Imat[:,t-1,al],
                                 Rmat[:,t-1,al], Inum, pred="t", thres=1e-2)
        caModel.weightSum()
        caModel.update()
        Amat[:,t,al], Smat[:,t,al], Imat[:,t,al], Rmat[:,t,al] = caModel.At, caModel.St, caModel.It, caModel.Rt

np.savetxt("Smat.csv", np.column_stack(Smat[:,i, 0] for i in range(time)), delimiter=",")
np.savetxt("Imat.csv", np.column_stack(Imat[:,i, 0] for i in range(time)), delimiter=",")
np.savetxt("Rmat.csv", np.column_stack(Rmat[:,i, 0] for i in range(time)), delimiter=",")

######################################################################################################
## risk prediction of travel
c_travel = list(['26161', '26163', '17031'])
fips0 = list(data['fips'])
t0=0

Itravel = CA_SIR.riskTravel(Imat[:,:,0], t0, c_travel, fips0)
np.savetxt("Itravel.csv", Itravel, delimiter=",")


########################################################################################################################
## make maps
fipsUSA = data['fips']
riskUSA = CA_SIR.riskFunc(Imat[:,:,0])
CA_SIR.plot_risk(fipsUSA, riskUSA, scope="usa", plotName="one_week_risk_USA")

indexNY = data.index[data['state'] == "NY"].tolist()
fipsNY = data['fips'][indexNY]
riskNY = riskUSA[indexNY]

CA_SIR.plot_risk(fipsNY, riskNY, scope="NY", plotName="one_week_risk_NY")

CA_SIR.plot_travel(data, Itravel, scope=["MI","IL"], plotName="travel")

