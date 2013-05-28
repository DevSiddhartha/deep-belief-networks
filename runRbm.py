import numpy as np
import rbm
import sys,pickle
from PIL import Image

def readImage(dfile):
  output = []
  for line in dfile:
    l = line.split()
    im = Image.open(l[0])

    imdata = im.getdata()
    row = list(imdata)
    rowbinary = [0 if(pixel >= 100) else 1 for pixel in row]
    output.append(rowbinary)

  data = np.array(output)
  return data

def readRBM():
  pkl_file = open(sys.argv[5], 'rb')
  r = pickle.load(pkl_file)
  return r
def readdata():

  dfile = open(sys.argv[1]);
  readrbm  = int(sys.argv[2])
  imageflag = int(sys.argv[3]);
  opt = int(sys.argv[4])

  if imageflag:
    opt = 0
    data = readImage(dfile)
  else:
    output = []
    for line in dfile:
      l = line.split();
      output.append([float(bit)for bit in l])
    data = np.array(output)
  return data,readrbm,imageflag,opt

def printRes(res):
  f = open(sys.argv[1]+'.res','w')
  for row in res:
    for c in row:
      f.write(str(c))
      f.write(' ')
    f.write('\n')
  f.close()

def saveRBM(r):
  f = open(sys.argv[1]+'.pkl','wb')
  pickle.dump(r,f,-1)
  f.close()

if __name__ == '__main__':
    data,readrbm,imageflag,opt = readdata()
    N,M = data.shape

    option = [0,0,0]
    option.insert(0,opt)

    dbn = rbm.DBN(M,[50,30,10,3],option)
    dbn.train(data,3000)
    res = dbn.run_hidden(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]]))
    if imageflag:
      img = Image.new("1",(50,60))
      k = 1
      for r in res:
        pixels = [ 255 if pixel == 0 else 0 for pixel in r ]
        img.putdata(pixels)
        img.save(sys.argv[1]+str(k)+'-guess.JPEG',"JPEG")
        k = k+1
    else:
      print res
