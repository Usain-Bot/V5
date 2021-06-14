#In this file: you can set all variables

#All pairs we want to get data from
PAIRS = ['BATUSDT', 'ETHUSDT', 'BNBUSDT', 'BTCUSDT', 'LTCUSDT']

#The time we want data from
SINCE = "750 days ago UTC"

#Where to store csv files
PATH_TO_STORAGE = "V5/Data"

#Wallet to start strategy
WALLET = 100

#Hours to test data (will cut data in 2)
HOURS_TO_TEST = 750

#Hours of features train before
STEPS = 150