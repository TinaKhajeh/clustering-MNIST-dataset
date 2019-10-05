def predictEM( datas ):
   #datas is N*K array
   import numpy as np
   predictY  = []
   for i in range(len(datas)):
      tmp = datas[i];
      maxindex = np.argmax(tmp)
      if (tmp[maxindex]==0):# when the clustering algorithm could not make decission
         predictY.append(-1)
      else:
         predictY.append(maxindex)
      
   return [predictY]