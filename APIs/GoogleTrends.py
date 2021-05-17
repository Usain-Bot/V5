import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import *
from pytrends.request import TrendReq

pytrends = TrendReq(hl='World',tz=60)


def GetData(currency=["bitcoin"], t1=datetime.now()):

    t2 = t1 - relativedelta(days=7)

    # build the payload
    #pytrends.build_payload(currency, timeframe='now 7-d')

    # store interest over time information in df
    #df1 = pytrends.interest_over_time()
    df2 = pytrends.get_historical_interest(
        currency, 
        year_start=int(t2.strftime("%Y")), 
        month_start=int(t2.strftime("%m")), 
        day_start=int(t2.strftime("%d")), 
        hour_start=int(t2.strftime("%H")), 
        year_end=int(t1.strftime("%Y")), 
        month_end=int(t1.strftime("%m")), 
        day_end=int(t1.strftime("%d")), 
        hour_end=int(t1.strftime("%H")), 
        cat=0, geo='', gprop='', sleep=0)

    # display the top 20 rows in dataframe
    #print(df2.head(20))

    # plot all three trends in same chart
    #df1.plot()
    #df2.plot()
    #plt.show()

    return (df2[currency[0]].values/100).tolist()