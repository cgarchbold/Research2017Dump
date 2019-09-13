import os
import sys
from random import randint

import shutil

for fname in os.listdir('C:/Users/Cohen/Desktop/train_panos'):
	for panoname in os.listdir('C:/Users/Cohen/Desktop/train_panos/'+fname):
		if randint(0, 5) <= 4:
			directory='C:/Users/Cohen/Desktop/newtrain_panos/'+fname
		else:
			directory='C:/Users/Cohen/Desktop/valid_panos/'+fname
		if not  os.path.exists(directory):
			os.makedirs(directory)
		shutil.copy('C:/Users/Cohen/Desktop/train_panos/'+fname+'/'+panoname, directory)