import subprocess
import os
import sys
from time import time
# 추후에 trading period도 조정 가능하도록 수정하기 (argument로 입력하도록 설정)

for filename in os.listdir('stockdatas'):
    start = time()

    # python run_binary_preprocessing.py 2880.TW 20 50
    # filename : 2880.TW_training.csv
    ticker = filename.split('_')[0]   # 2880.TW

    #if os.path.exists('dataset2/20_50/' + ticker):
    #  print('dataset ' + ticker +' already exists')
    #  continue
    # exception : ^BSESN_testing.csv
    split_ticker = ticker.split('.')
    if len(split_ticker) != 2:
      continue

    # only read the name of training data file
    training = filename.split('_')[1]
    if training != 'training.csv':
      continue

    # only read TW, tw, JK
    country = split_ticker[1]
    if country != 'TW' and country != 'tw' and country != 'JK':
      continue
    
    subprocess.call(f'python run_binary_preprocessing.py {ticker} 20 50', shell=True)
    
    print("\n*** Running default preprocessing is completed. ***\n")

    #===========================================================================

    ticker_without_dot = ticker.replace('.', '')     # 2880TW
    subprocess.call(f'python generatedata.py dataset2 20_50/{ticker} dataset_{ticker_without_dot}_20_50', shell=True)
    
    print("\n*** Running dataset generating is completed. ***")
    end = time()
    print("*** Time for generating a dataset for one ticker :", round(end-start, 1), "\n")
    #break