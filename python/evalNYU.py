# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.

#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import numpy as np
#import cv2
import scipy.io
import subprocess as subp

import os, re, time, random
import argparse

#from vpCluster.rgbd.rgbdframe import RgbdFrame
#from vpCluster.manifold.sphere import Sphere
#from js.utils.config import Config2String
#from js.utils.plot.pyplot import SaveFigureAsImage

def run(cfg,reRun):
  #args = ['../build/dpSubclusterSphereGMM',
  #args = ['../build/dpStickGMM',
  print 'processing '+cfg['dataPath']+cfg['filePath']
  print "output to "+cfg['outName']
  args = ['../pod-build/bin/realtimeMF',
    '--mode '+cfg['mode'],
    '-i {}'.format(cfg['dataPath']+cfg['filePath']+"_d.png"),
    '-o {}'.format(cfg['outName']),
    '-B {}'.format(5),
    '-T {}'.format(30),
    ]
  if 'dt' in cfg.keys():
    args.append('--dt {}'.format(cfg['dt']))
  if 'tMax' in cfg.keys():
    args.append('--tMax {}'.format(cfg['tMax']))
  if 'nCGIter' in cfg.keys():
    args.append('--nCGIter {}'.format(cfg['nCGIter']))

  print "checking if " + cfg['outName']+"_cRmf.csv"
  if reRun or not os.path.isfile(cfg['outName']+"_cRmf.csv"):
    print ' '.join(args)
    print ' --------------------- '
    time.sleep(1)
    err = subp.call(' '.join(args),shell=True)
    if err:
      print 'error when executing'
  else:
    print "skipping " + cfg['dataPath']+cfg['filePath']
#      raw_input()
#  z = np.loadtxt(cfg['outName']+'.lbl',dtype=int,delimiter=' ')
#  sil = np.loadtxt(cfg['outName']+'.lbl_measures.csv',delimiter=" ")


def config2Str(cfg):
  use = ['mode','dt','tMax','nCGIter']
  st = use[0]+'_'+str(cfg[use[0]])
  for key in use[1::]:
    if key in cfg.keys():
      st += '-'+key+'_'+str(cfg[key])
  return st

parser = argparse.ArgumentParser(description = 'rtmf extraction for NYU')
parser.add_argument('-m','--mode', default='vmf', 
    help='vmf, approx, direct')
args = parser.parse_args()

cfg=dict()
cfg['mode'] = args.mode;
cfg['resultsPath'] = '/data/vision/scratch/fisher/jstraub/rtmf/nyu/'
cfg['dataPath'] = "/data/vision/fisher/data1/nyu_depth_v2/extracted/"

cfg['resultsPath'] = './'
cfg['dataPath'] = "../data/"

# for eval of the high quality results of the direct method
cfg['nCGIter'] = 25
cfg['dt'] = 0.05
cfg['tMax'] = 5.0

reRun = True
printCmd = True


names = []
for root, dirs, files in os.walk(cfg["dataPath"]):
  for file in files:
    name,ending = os.path.splitext(file)
    if ending == '.png' and not re.search("_rgb",name) is None:
      names.append(re.sub("_rgb","",name))
  break
random.shuffle(names)

error = np.zeros((2,len(names)))
for i,name in enumerate(names):
  cfg['filePath'] = name
  cfg['outName'] = cfg['resultsPath']+cfg['filePath']+'_'+config2Str(cfg)
  run(cfg,reRun)


