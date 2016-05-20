import numpy as np
import os.path, re, sys
import scipy.io as scio
from scipy.linalg import det
import cv2
import itertools
from js.data.rgbd.rgbdframe import *
import mayavi.mlab as mlab
import matplotlib.pyplot as plt

def plotMF(fig,R,col=None):
  mfColor = []
  mfColor.append((232/255.0,65/255.0,32/255.0)) # red
  mfColor.append((32/255.0,232/255.0,59/255.0)) # green
  mfColor.append((32/255.0,182/255.0,232/255.0)) # tuerkis
  pts = np.zeros((3,6))
  for i in range(0,3):
    pts[:,i*2]  = -R[:,i] 
    pts[:,i*2+1] = R[:,i] 
  if col is None:
    mlab.plot3d(pts[0,0:2],pts[1,0:2],pts[2,0:2],figure=fig,color=mfColor[0])
    mlab.plot3d(pts[0,2:4],pts[1,2:4],pts[2,2:4],figure=fig,color=mfColor[1])
    mlab.plot3d(pts[0,4:6],pts[1,4:6],pts[2,4:6],figure=fig,color=mfColor[2])
  else:
    mlab.plot3d(pts[0,0:2],pts[1,0:2],pts[2,0:2],figure=fig,color=col)
    mlab.plot3d(pts[0,2:4],pts[1,2:4],pts[2,2:4],figure=fig,color=col)
    mlab.plot3d(pts[0,4:6],pts[1,4:6],pts[2,4:6],figure=fig,color=col)

def ExtractObjectDirections(lImg, nImg, lId):
  errosionSize=8
  labelMap = ((lImg==lId)*255).astype(np.uint8)
  kernel = np.ones((errosionSize, errosionSize),np.uint8)
  labelMapE = cv2.erode(labelMap,kernel,iterations=1)
#  plt.imshow(np.concatenate((labelMap,labelMapE),axis=1))
#  plt.show()
#  print lImg.shape
#  print nImg.shape
  nLabel = nImg[labelMapE>128,:].T
#  print nLabel.shape, np.isnan(nLabel).sum()
  nLabel = nLabel[:,np.logical_not(np.isnan(nLabel[0,:]))]
  return nLabel
def ExtractObjectDirection(lImg, nImg, lId=11):
  errosionSize=8
  labelMap = ((lImg==lId)*255).astype(np.uint8)
  kernel = np.ones((errosionSize, errosionSize),np.uint8)
  labelMapE = cv2.erode(labelMap,kernel,iterations=1)
#  plt.imshow(np.concatenate((labelMap,labelMapE),axis=1))
#  plt.show()
#  print lImg.shape
#  print nImg.shape
  nLabel = nImg[labelMapE>128,:].T
#  print nLabel.shape, np.isnan(nLabel).sum()
  nLabel = nLabel[:,np.logical_not(np.isnan(nLabel[0,:]))]
#  print nLabel.shape, np.isnan(nLabel).sum()
  nMean = nLabel.sum(axis=1)
  nMean /= np.sqrt((nMean**2).sum())
  return nMean

mode = "directGD"
mode = "approx"
mode = "approxGD"
mode = "vmfCF"
mode = "vmf"
mode = "direct"
mode = "mmfvmf"
nyuPath = "/data/vision/fisher/data1/nyu_depth_v2/"
rtmfPath = "/data/vision/scratch/fisher/jstraub/rtmf/nyu/"

with open(os.path.join(nyuPath,"labels.txt")) as f:
  labels = [label[:-1] for label in f.readlines()]
print len(labels), labels

if True  and os.path.isfile("./angularObjectDeviations_rtmf_"+mode+".csv"):
  error = np.loadtxt("./angularObjectDeviations_rtmf_"+mode+".csv")
  plt.figure()
  percentiles = []
  percentiles.append([])
  percentiles.append([])
  percentiles.append([])
  errorsSorted = []
  dispLabels = []
  for i in range(1,len(labels)):
#    print "nans: ", np.isnan(error[i,:]).sum(), "of", error[i,:].size
    idOk = np.logical_not(np.isnan(error[i,:]))
    if idOk.sum() > 500:
      errorS = error[i,idOk].copy().tolist()
      errorS.sort()
      errorsSorted.append(errorS)
      percentiles[0].append(errorS[int(np.floor(0.25*len(errorS)))])
      percentiles[1].append(errorS[int(np.floor(0.5*len(errorS)))])
      percentiles[2].append(errorS[int(np.floor(0.75*len(errorS)))])
      dispLabels.append(labels[i-1]+"{}".format(i))
#      print i, dispLabels[-1], idOk.sum(), percentiles[0][-1], percentiles[1][-1], percentiles[2][-1]
  idSort = np.argsort(np.array(percentiles[1]))
  N = min(40, idSort.size)
  plt.plot(np.arange(N),np.array(percentiles[0])[idSort[:N]],'o')
  plt.plot(np.arange(N),np.array(percentiles[1])[idSort[:N]],'o')
  plt.plot(np.arange(N),np.array(percentiles[2])[idSort[:N]],'o')
  plt.xticks(np.arange(N),[dispLabels[i] for i in idSort[:N]],rotation='vertical')
#      plt.plot(errorS,1.*np.arange(len(errorS))/(len(errorS)-1),label=labels[i])
  print [(dispLabels[i], percentiles[0][i],percentiles[1][i],percentiles[2][i]) for i in idSort]
  plt.ylim([0,45])
#  plt.xlim([0,50])
#  plt.legend(loc="best")
  plt.grid(True)
  plt.ylabel("angular error at percentile") 

  plt.figure()
  for i,errorS in enumerate(errorsSorted):
    plt.plot(errorS,1.*np.arange(len(errorS))/(len(errorS)-1),label=dispLabels[i])
  plt.ylim([0,1])
  plt.xlim([0,50])
  plt.legend(loc="best")
  plt.grid(True)
  plt.ylabel("percentile") 
  plt.show()

if os.path.isfile("./rtmfPaths_"+mode+".txt"):
  with open("./rtmfPaths_"+mode+".txt","r") as f:
    rtmfPaths = [path[:-1] for path in f.readlines()]
else:
  rtmfPaths = []
  for root, dirs, files in os.walk(rtmfPath):
    for f in files:
      if re.search("[a-z_]+_[0-9]+_[0-9]+_mode_"+mode+"-[-_.0-9a-zA-Z]+_cRmf.csv", f):
        rtmfPaths.append(os.path.join(root,f))
  rtmfPaths.sort()
  with open("./rtmfPaths_"+mode+".txt","w") as f:
    f.writelines([path+"\n" for path in rtmfPaths])
print len(rtmfPaths)

labelImgPaths = []
for root, dirs, files in os.walk(nyuPath):
  for f in files:
    if re.search("[a-z_]+_[0-9]+_[0-9]+_l.png", f):
      labelImgPaths.append(os.path.join(root,f))
labelImgPaths.sort()
print len(labelImgPaths)

#import matplotlib.pyplot as plt
#plt.figure()
error = np.zeros((len(labels),len(rtmfPaths)))
for i,rtmfPath in enumerate(rtmfPaths):
  rtmfName = re.sub("_mode_"+mode+"-[-_.0-9a-zA-Z]+_cRmf.csv","",os.path.split(rtmfPath)[1]) 
  labelImgPathMatch = ""
  for labelImgPath in labelImgPaths:
    labelName = re.sub("_l.png","",os.path.split(labelImgPath)[1])
    if labelName == rtmfName:
      labelImgPathMatch = labelImgPath
      break
  labelName = re.sub("_l.png","",os.path.split(labelImgPathMatch)[1])
  if not rtmfName == labelName:
    print " !!!!!!!!!!!! "
    print os.path.split(rtmfPath)[1], rtmfName
    print os.path.split(labelImgPathMatch)[1], labelName
    raw_input()
    continue
#  try:
  R = np.loadtxt(rtmfPath)
  rgbd = RgbdFrame(540.)
  rgbd.load(re.sub("_l.png","",labelImgPathMatch ))
  lImg = cv2.imread(labelImgPathMatch,cv2.CV_LOAD_IMAGE_UNCHANGED)
  nImg = rgbd.getNormals().copy()
  print "R_rtmf", R
  for lId in range(len(labels)):
    ns = ExtractObjectDirections(lImg,nImg,lId+1)
    M = np.concatenate((R, -R),axis=1)
#    anglesToY = []
#    for ids in itertools.combinations(np.arange(6),3):
#      Rc = np.zeros((3,3))
#      for l in range(3):
#        Rc[:,l] = M[:,ids[l]]
#      if det(Rc) > 0:
#        Rn = Rc.T.dot(nMean)
#        anglesToY.append(np.arccos(np.abs(Rn[1]))*180./np.pi)
##        print anglesToY[-1], Rn
#    error[lId,i] = min(anglesToY)
    error[lId,i]=np.mean(np.arccos(np.max(np.abs(M.T.dot(ns)),axis=0))*180./np.pi)
    print "direction of {} surface normals:".format(labels[lId])," error ",error[lId,i]

  if False:
    n = rgbd.getNormals()[rgbd.mask,:]
    figm = mlab.figure(bgcolor=(1,1,1))
    mlab.points3d(n[:,0],n[:,1],n[:,2], color=(0.5,0.5,0.5),mode="point")
    plotMF(figm,R)
    mlab.show(stop=True)
  
#  except:
#    print "Unexpected error:", sys.exc_info()[0]
#    error[i] = np.nan
np.savetxt("./angularObjectDeviations_rtmf_"+mode+".csv",error)

