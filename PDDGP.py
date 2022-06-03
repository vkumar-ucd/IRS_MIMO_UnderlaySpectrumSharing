# Achievable rate maximization for underlay spectrum sharing MIMO system with intelligent reflecting surface
# Authors: (*)Vaibhav Kumar, (*)Mark F. Flanagan, (^)Rui Zhang, and (*)Le-Nam Tran
# (*): School of Electrical and Electronic Engineering, University College Dublin, Ireland
# (^): Department of Electrical and Computer Engineering, National University of Singapore, Singapore
# email: vaibhav.kumar@ucdconnect.ie / vaibhav.kumar@ucd.ie / vaibhav.kumar@ieee.org

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)  # seed for random number generation

# Function for Hermitian
def Herm(x):
    return x.conj().T

# Function for dB to power conversion
def db2pow(x):
    return 10**(0.1*x)

# Function for power to dB conversion
def pow2db(x):
    return 10*np.log10(x)       

# Function to generate channel fading coefficients
def ChanGen(zeta,d,dim1,dim2):
    d0 = 1 # reference distance for pathloss
    PL = db2pow(-30-10*zeta*np.log10(d/d0)) # PL at a distance of d meters
    y = np.sqrt(PL/2)*(np.random.randn(dim1,dim2)+1j*np.random.randn(dim1,dim2))
    return y

# Function to calculate the augmented objective
def AugObjCalc(X,theta):
    THETA = np.diag(theta)                                      # diagonal IRS phase-shift matrix
    zR = H_IR@THETA@H_TI+H_TR                                   # effective ST-SR channel
    zk = H_Ik@THETA@H_TI+H_Tk                                   # effective ST-PR channel
    gk = (np.trace((zk@X@(zk.conj().transpose(0,2,1))),\
            axis1=1,axis2=2))+sVec-Pk                      # g_k() function defined on page 3
    y = np.real(np.log(np.linalg.det(np.eye(Nr)+zR@X@zR.conj().T))\
        -np.sum(upsilonVec*gk)-0.5*rho*np.sum((gk**2)))    # the augmented objective on page 3
    return y

# Function to calculate the (true) objective in nats/s/Hz
def ObjCalc(X,theta):
    THETA = np.diagflat(np.matrix(theta))                       # diagonal IRS phase-shift matrix
    zR = H_IR @ THETA @ H_TI + H_TR                             # effective ST-SR channel
    y = np.real(np.log(np.linalg.det(np.eye(Nr)+\
                zR@X@zR.conj().T)))                             # SR achievable rate in nats/s/Hz
    return y    

# Function to update theta vector
def thetaUpdate(X,thetaOld):
    grad = gradThetaCalc(X,thetaOld)    # gradient of augmented objective wrt theta vector   
    stepSize = stepTheta                                # local variable assignment
    thetaNew = thetaOld+(1/stepSize)*grad               # updated theta
    thetaNew = thetaNew/abs(thetaNew)                   # projecting thetaNew onto feasible set for theta
    # line search to find stepSize for updating theta
    while AugObjCalc(X,thetaNew) < QLTheta(X,thetaOld,thetaNew,grad,stepSize):
        stepSize = 2*stepSize
        thetaNew = thetaOld + (1/stepSize)*grad         # updated theta
        thetaNew = thetaNew/abs(thetaNew)               # projecting thetaNew onto feasible set for theta
    stepSize = np.maximum(stepThreshold,stepSize/2)
    return stepSize,thetaNew   

# Function for gradient wrt theta vector (refer to (5) in the paper)
def gradThetaCalc(X,theta):
    THETA = np.diag(theta)          # diagonal IRS phase-shift matrix
    zR = H_IR@THETA@H_TI+H_TR       # effective ST-SR channel
    zk = H_Ik@THETA@H_TI+H_Tk       # effective ST-PR channel
    term1 = np.diag(Herm(H_IR)@np.linalg.inv(np.eye(Nr)+zR@X@Herm(zR))@zR@X@Herm(H_TI))
    term2 = upsilonVec+rho*((np.trace((zk@X@(zk.conj().transpose(0,2,1))),axis1=1,axis2=2))+sVec-Pk)
    term3 = np.diagonal((H_Ik.conj().transpose(0,2,1))@zk@X@Herm(H_TI),\
                        axis1=1, axis2=2)
    y = term1 - (term2[:,None]*term3).sum(axis=0) 
    return np.transpose(y)      

# Function for QLTheta for line search 
def QLTheta(X,thetaOld,thetaNew,grad,stepSize):
    y = np.real(AugObjCalc(X,thetaOld) \
        + 2*np.real(Herm(grad)@(thetaNew-thetaOld))\
        -0.5*stepSize*((np.linalg.norm(thetaNew-thetaOld))**2))
    return y   

# Function to update X matrix
def XUpdate(XOld,theta):
    grad = gradXCalc(XOld,theta)            # gradient of augmented objective wrt X 
    stepSize = stepX                        # local variable assignment  
    XNew = projX(XOld,grad,stepSize)        # updating X and projecting onto the feasible set of X
    # line search to find stepSize for updating X
    while AugObjCalc(XNew,theta) < QLX(XNew,XOld,theta,grad,stepSize):
        stepSize = 2*stepSize               # update the step size 
        XNew = projX(XOld,grad,stepSize)    # updating X and projecting onto the feasible set of X
    stepSize = np.maximum(stepThreshold,stepSize/2)    
    return stepSize,XNew,grad

# Function for gradient wrt X matrix (refer to (7) in the paper)
def gradXCalc(X,theta):
    THETA = np.diag(theta)          # diagonal IRS phase-shift matrix
    zR = H_IR@THETA@H_TI+H_TR       # effective ST-SR channel      
    zk = H_Ik@THETA@H_TI+H_Tk       # effective ST-PR channel
    term1 = Herm(zR)@np.linalg.inv(np.eye(Nr)+zR@X@Herm(zR))@zR
    term2 = upsilonVec+rho*((np.trace((zk@X@(zk.conj().transpose(0,2,1))),axis1=1,axis2=2))+sVec-Pk)
    term3 = (zk.conj().transpose(0,2,1))@zk
    y = term1 - (term2[:,None,None]*term3).sum(axis=0)
    return y

# Projection onto the feasible set of X (Water-Filling solution)
def projX(X,grad,stepSize):
    W = X+(1/stepSize)*grad
    eigval, eigvec = np.linalg.eig(W)
    EigVal = np.real(eigval)
    if np.sum(np.maximum(EigVal,0)) < Pmax:
        EIGVAL = np.maximum(EigVal,0)
    else:
        EIGVAL = projsplx(EigVal/Pmax)*Pmax
    y = np.array(eigvec @ np.diagflat(np.matrix(EIGVAL))@(eigvec.conj().T))
    return y

# Function for simplex projection
def projsplx(v):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0)
    return w

# Function for QLX
def QLX(xNew,xOld,theta,grad,stepSize):
    y = np.real(AugObjCalc(xOld,theta)+np.trace(grad@(xNew-xOld))\
                - 0.5*stepSize*((np.linalg.norm(xNew-xOld,'fro'))**2))
    return y
        

###############
#
#
#-------- Main function
#
#
###############      

# Power and interference budgets 
Pmax = db2pow(20-30)                # transmit power budget of 20 dBm
B = 10e6                            # bandwidth is 10 MHz
No = db2pow(-174-30)                # noise PSD of -174 dBm/Hz
NoisePower = B*No                   # noise power
Pk  = (1e-13)/NoisePower            # normalized interference tolerance at PRs

# Number of antennas and tiles 
Nt = 4                              # number of ST antennas
Nr = 4                              # number of SR antennas
Np = 4                              # number of PR antennas
Ni = 64                             # number of IRS tiles
K = 4                               # number of PRs

# Path loss exponents 
zetaDirect = 3.75                   # PL exponent for direct links
zetaRIS = 2.2                       # PL exponent for RIS-related links

# Location of different nodes
locP = np.zeros((K,2))                          # memory initialization
locP[:,1] = np.array([5*k for k in range(K)])   # PR locations in 2D
locT = np.array([300, 0])                       # ST location in 2D
locI = np.array([300, 30])                      # IRS location in 2D
locR = np.array([600, 0])                       # SR location in 2D

locPNew = np.zeros((K,2))
locPNew[:,1] = np.array([5*k for k in range(K)])

# Distance between different nodes
dTI = np.linalg.norm(locT - locI)                                   # ST-IRS distance
dTR = np.linalg.norm(locT - locR)                                   # ST-SR distance
dIR = np.linalg.norm(locI - locR)                                   # IRS-SR distance 
dTk = np.array([np.linalg.norm(locT-locP[k,:]) for k in range(K)])  # ST-PR distance
dIk = np.array([np.linalg.norm(locI-locP[k,:]) for k in range(K)])  # IRS-PR distance

# Channel generation
NormFact = 1/np.sqrt(NoisePower)                                                # normalization factor
H_TI = NormFact*ChanGen(zetaRIS, dTI, Ni, Nt)                                   # ST-IRS channel
H_TR = NormFact*ChanGen(zetaDirect, dTR, Nr, Nt)                                # ST-SR channel
H_IR = ChanGen(zetaRIS, dIR, Nr, Ni)                                            # IRS-SR channel
H_Tk = np.array([NormFact*ChanGen(zetaDirect,dTk[k],Np,Nt) for k in range(K)])  # ST-PR channel
H_Ik = np.array([ChanGen(zetaRIS,dIk[k],Np,Ni) for k in range(K)])              # IRS-PR channel

# The PDDGP algorithm
stepThreshold = 0.01                                            # threshold for step size
stepX = 0.01                                                    # step size for X
stepTheta = 0.01                                                # step size for Theta
XOld = np.zeros((Nt,Nt), dtype = 'complex_')                    # random covariance matrix
thetaVecOld = np.exp(1j*2*np.pi*np.random.rand(Ni,))            # random passive beamformer
sVec = np.zeros((K,))                                           # initialization for the s vector 
upsilonVec = np.zeros(np.shape(sVec))                           # initialization for the upsilon vector
rho = 0.1                                                       # exterior penalty parameter
Zk = np.zeros(np.shape(H_Tk), dtype = 'complex_')               # memory allocation
ObjSeq = np.array([])                                           # array initialization to store true objective
AugObjSeq = np.array([])                                        # array initialization to store augmented objective
iIter = -1                                                      # counter initialization
AugObjDiff = 1e3                                                # arbirtary large number
ObjsDiff = 1e3                                                  # arbitrary large number                                     
epsilon = 1e-5                                                  # convergene tolerance

while (AugObjDiff > epsilon) or (ObjsDiff > epsilon):

    iIter = iIter + 1   # counter update

    #-------- line 2 in Algorithm 2
    # updating theta
    stepTheta,thetaVecOld = thetaUpdate(XOld,thetaVecOld)   # line 3 in Algorithm

    # updating X
    stepX,XOld,gradX = XUpdate(XOld,thetaVecOld)            # line 4 in Algorithm 1    
    
    # updating s vector 
    Zk = H_Ik@(np.diag(thetaVecOld))@H_TI+H_Tk              # line 6 in Algorithm 1
    sVec = np.maximum(0, np.real(Pk-(np.trace((Zk@XOld@(Zk.conj().transpose(0,2,1))),axis1=1,axis2=2))))
                                                            # line 5 in Algorithm 1

    #--------- calculating the (true) objective and augmented objective      
    ObjSeq = np.append(ObjSeq,np.log2(np.exp(1))*ObjCalc(XOld, thetaVecOld))            # ture objective in bps/s/Hz
    AugObjSeq = np.append(AugObjSeq, np.log2(np.exp(1))*AugObjCalc(XOld,thetaVecOld))   # augmented objective    
    if iIter > 2:
        AugObjDiff = abs(AugObjSeq[iIter] - AugObjSeq[iIter-1])/AugObjSeq[iIter-1]
        if AugObjDiff <= epsilon:                                           # first condition for convergence
            upsilonVec = upsilonVec+rho*np.real(
                            (np.trace((Zk@XOld@(Zk.conj().transpose(0,2,1))),axis1=1,axis2=2)\
                            +sVec-Pk))          # line 4 in Algorithm 2
            rho = 10*rho                        # line 5 in Algorithm 2
    ObjsDiff = abs(AugObjSeq[iIter] - ObjSeq[iIter])/AugObjSeq[iIter]       # second condition for convergence 

# Checking the feasibility of the solution
print('===========================')
print(f"Pmax = {pow2db(Pmax)+30:.2f} dBm")
print(f"trace(X) = {pow2db(np.real(np.trace(XOld)))+30:.2f} dBm")
print(f"Maximum tolerable interference = {pow2db(Pk)+30:.2f} dBm")
for k in range(K):
    print(f"Interfernce at PR{k} = {pow2db(np.real(np.trace(Zk[k,:,:]@XOld@(Zk[k,:,:].conj().T))))+30:.2f} dBm")
print('===========================')

# Plotting the figure
plt.plot(np.arange(iIter+1),ObjSeq,'k-',label=r'$R(\mathbf{X},\mathbf{\theta})$')
plt.plot(np.arange(iIter+1),AugObjSeq,'k--',label=r'$\hat{R}_{\mathbf{\upsilon},\rho}(\mathbf{X},\mathbf{\theta})$')
plt.xlabel('Iteration number',fontsize=20)
plt.ylabel('Rate (bps/Hz)',fontsize=20)
plt.legend(fontsize=20)
plt.show()