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

def ExtractFloorDirection(pathLabelImage, nImg, lFloor=11):
  errosionSize=8
  print pathLabelImage
  L = cv2.imread(pathLabelImage,cv2.CV_LOAD_IMAGE_UNCHANGED)
  floorMap = ((L==lFloor)*255).astype(np.uint8)
  kernel = np.ones((errosionSize, errosionSize),np.uint8)
  floorMapE = cv2.erode(floorMap,kernel,iterations=1)
#  plt.imshow(np.concatenate((floorMap,floorMapE),axis=1))
#  plt.show()
  print L.shape
  print nImg.shape
  nFloor = nImg[floorMapE>128,:].T
  print nFloor.shape, np.isnan(nFloor).sum()
  nFloor = nFloor[:,np.logical_not(np.isnan(nFloor[0,:]))]
  print nFloor.shape, np.isnan(nFloor).sum()
  nMean = nFloor.sum(axis=1)
  nMean /= np.sqrt((nMean**2).sum())
  return nMean

mode = "approx"
mode = "vmf"
mode = "vmfCF"
mode = "approxGD"
nyuPath = "/data/vision/fisher/data1/nyu_depth_v2/"
rtmfPath = "/data/vision/scratch/fisher/jstraub/rtmf/nyu/"

if True  and os.path.isfile("./angularFloorDeviations_rtmf_"+mode+".csv"):
  error = np.loadtxt("./angularFloorDeviations_rtmf_"+mode+".csv")
  print "nans: ", np.isnan(error[1,:]).sum(), "of", error.size
  error = error[:,np.logical_not(np.isnan(error[1,:]))]
  print error.shape
  labels = ["unaligned","RTMF "+mode]
  plt.figure()
  for i in range(2):
    errorS = error[i,:].tolist()
    errorS.sort()
    plt.plot(errorS,1.*np.arange(len(errorS))/(len(errorS)-1),label=labels[i])
    plt.ylim([0,1])
    plt.xlim([0,25])
  plt.legend(loc="best")
  plt.ylabel("precentage of scenes")
  plt.xlabel("degrees from vertical")
  plt.grid(True)
  plt.show()

with open(os.path.join(nyuPath,"labels.txt")) as f:
  labels = [label[:-1] for label in f.readlines()]
print labels[:20]
lFloor = 0
for i,label in enumerate(labels):
  if label == "floor":
    lFloor = i+1
    break
print "label of floor: ", lFloor

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
error = np.zeros((2,len(rtmfPaths)))
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
  nMean = ExtractFloorDirection(labelImgPathMatch,rgbd.getNormals())
  error[0,i] = np.arccos(np.abs(nMean[1]))*180./np.pi
  print "direction of floor surface normals: ", nMean
  print "R_rtmf", R
  pcC = rgbd.getPc()[rgbd.mask,:].T
  anglesToY = []
  M = np.concatenate((R, -R),axis=1)
#    print M
  for ids in itertools.combinations(np.arange(6),3):
    Rc = np.zeros((3,3))
    for l in range(3):
      Rc[:,l] = M[:,ids[l]]
    if det(Rc) > 0:
      Rn = Rc.T.dot(nMean)
      anglesToY.append(np.arccos(np.abs(Rn[1]))*180./np.pi)
      print anglesToY[-1], Rn
#        figm = mlab.figure(bgcolor=(1,1,1))
#        pc = Rc.T.dot(pcC)
#        mlab.points3d(pc[0,:],pc[1,:],pc[2,:],
#            rgbd.gray[rgbd.mask],colormap='gray',scale_factor=0.01,
#            figure=figm,mode='point',mask_points=1)
#        mlab.show(stop=True)
#        mlab.close(figm)
  error[1,i] = min(anglesToY)
  print error[:,i]

  if False:
    n = rgbd.getNormals()[rgbd.mask,:]
    figm = mlab.figure(bgcolor=(1,1,1))
    mlab.points3d(n[:,0],n[:,1],n[:,2], color=(0.5,0.5,0.5),mode="point")
    plotMF(figm,R)
    mlab.show(stop=True)
  
#  except:
#    print "Unexpected error:", sys.exc_info()[0]
#    error[i] = np.nan
np.savetxt("./angularFloorDeviations_rtmf_"+mode+".csv",error)

