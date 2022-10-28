# test.py
import sys
import os
sys.path.append(os.path.abspath('DL'))
curdir = os.getcwd()
os.chdir('DL')
import train
a = train.train()
os.chdir(curdir)
print('end')