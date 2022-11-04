import numpy as np
import random
np.random.seed(3)
random.seed(5)

#prng2 = np.random.RandomState(5)

#print(prng2.uniform(-1,1))

for i in range(5):
   print(i, random.choice([1,2,3]))
for i in range(10):
   #prng2.uniform(-1,1)
   print(i, np.random.uniform(-1,1))


