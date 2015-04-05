# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.
import numpy as np


BLOCK_SIZE = 256
mu = np.ones(BLOCK_SIZE*4*6) #list(BLOCK_SIZE*4*6)
tpr = BLOCK_SIZE/(4*6); # threads per row
print 'tpr=',tpr
#reduction.....
print "__syncthreads();" #sync the threads

for tid in [0,1,2,3,4,5,6,7,8,9]:#range(0,1):
  print 'tid=',tid
  for r in range(4*6): #(r=0; r<4*6; ++r)
    if r*tpr <= tid and tid < (r+1)*tpr:
      #print 'r=',r,' tpr=',tpr
      tidr = tid - r*tpr; # id in row
      #print 'tidr=',tidr
      for s in [64]:#,8,4,2,1]:# (s=(BLOCK_SIZE)/2; s>0; s>>=1):
      #for s in [128,64,32,16,8,4,2,1]:# (s=(BLOCK_SIZE)/2; s>0; s>>=1):
        expr = s/tpr; # executions per row
        print 'expr=',expr,' s=',s,' tpr=',tpr
        for ex in range(expr): #(ex=0; ex<expr; ++ex):
          #print 'ex=',ex,' tidr=',tidr
          print "add {},{} and {},{}".format(r,tidr+ex*tpr, r,tidr+ex*tpr+s)
          mu[r*BLOCK_SIZE+tidr+ex*tpr] += mu[r*BLOCK_SIZE+tidr+ex*tpr+s];
          if tidr+ex*tpr+s >= 2*s:
            print "ERRROR"
        exprem = s%tpr; # remaining executions
        #print 'exprem=',exprem
        if (tidr <exprem):
          print "add remaining {},{} and {},{}".format(r,expr*tpr+tidr, r,tidr+expr*tpr+s)
          mu[r*BLOCK_SIZE+tidr+expr*tpr] += mu[r*BLOCK_SIZE+tidr+expr*tpr+s];
        print "__syncthreads();" #sync the threads
  for k in range(6*4): #(k=0; k<6*4; ++k):
    if(tid==k): #  && Ni[(k/3)*BLOCK_SIZE]>0 ) {
      print mu[k*BLOCK_SIZE]
      #atomicAdd(&mu_karch[k],mu[k*BLOCK_SIZE]);

print mu[0:255]

print 'sums'
for k in range(6*4): #(k=0; k<6*4; ++k):
  print mu[k*BLOCK_SIZE]
