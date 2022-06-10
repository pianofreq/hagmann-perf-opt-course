import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time
import pytz
import sys
from IPython.display import display, clear_output

class BollTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, SMA, dev, units):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []
        
        #*****************add strategy-specific attributes here******************
        self.SMA = SMA
        self.dev = dev
        #************************************************************************
    
    def get_most_recent(self, days = 5):
        while True:
            time.sleep(2)
            now = datetime.utcnow()
            now = now - timedelta(microseconds = now.microsecond)
            past = now - timedelta(days = days)
            df = self.get_history(instrument = self.instrument, start = past, end = now,
                                   granularity = "S5", price = "M", localize = False).c.dropna().to_frame()
            df.rename(columns = {"c":self.instrument}, inplace = True)
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
                break
                
    def on_success(self, time, bid, ask):
        if self.ticks > 1:
            sys.stdout.write("DATA POINT #"+str(len(self.tick_data))+": "+str(self.tick_data.tail(1))+"\n\n")
        
        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument:(ask + bid)/2}, 
                          index = [recent_tick])
        self.tick_data = self.tick_data.append(df)
        
        if recent_tick - self.last_bar > self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()
    
    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                                  label="right").last().ffill().iloc[:-1])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
    
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** define your strategy here ************************
        df["SMA"] = df[self.instrument].rolling(self.SMA).mean()
        df["Lower"] = df["SMA"] - df[self.instrument].rolling(self.SMA).std() * self.dev
        df["Upper"] = df["SMA"] + df[self.instrument].rolling(self.SMA).std() * self.dev
        df["distance"] = df[self.instrument] - df.SMA
        df["position"] = np.where(df[self.instrument] < df.Lower, 1, np.nan)
        df["position"] = np.where(df[self.instrument] > df.Upper, -1, df["position"])
        df["position"] = np.where(df.distance * df.distance.shift(1) < 0, 0, df["position"])
        df["position"] = df.position.ffill().fillna(0)
        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        sys.stdout.write("\n")
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING LONG")  # NEW
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
                self.report_trade(order, "GOING LONG")  # NEW
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")  # NEW
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")  # NEW
            self.position = -1
        elif self.data["position"].iloc[-1] == 0: 
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")  # NEW
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING NEUTRAL")  # NEW
            self.position = 0
    
    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        sys.stdout.write(110* "-"+"\n")
        sys.stdout.write("{} | {}\n\n".format(time, going))
        sys.stdout.write("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}\n".format(time, units, price, pl, cumpl))
        sys.stdout.write(110 * "-" + "\n")
        
        
if __name__ == "__main__":
    paper_EURAUD = BollTrader('oanda.cfg', "EUR_AUD", "25min", SMA = 121, dev = 2.8181818181818183, units = 100000)
    paper_EURAUD.get_most_recent()
    paper_EURAUD.stream_data(paper_EURAUD.instrument)#, stop = 20)
    if paper_EURAUD.position != 0: # if we have a final open position
        close_order = paper_EURAUD.create_order(paper_EURAUD.instrument,\
        units = -paper_EURAUD.position * paper_EURAUD.units, suppress = True, ret = True) 
        paper_EURAUD.report_trade(close_order, "GOING NEUTRAL")
        paper_EURAUD.position = 0
#====================================================================================================
#BEST COMBINATION: FREQUENCY = 25.0-MINUTE CHART | SMA = 121.0 | DEV = 2.8181818181818183 | Multiple = 1.192359
#----------------------------------------------------------------------------------------------------
#====================================================================================================
#SIMPLE BOLL STRATEGY | INSTRUMENT = EURAUD | FREQ = 25.0 | SMA = 121 | DEV = 2.8181818181818183 | LEV: 19.62
#----------------------------------------------------------------------------------------------------


#PERFORMANCE MEASURES:


#Multiple (Strategy):         15.827819
#Multiple (Buy-and-Hold):     1.044632
#--------------------------------------
#Out-/Underperformance:       14.783187


#CAGR:                        3.051468
#Annualized Mean:             1.396896
#Annualized Std:              0.985815
#Sharpe Ratio:                1.416996
#Sortino Ratio:               2.010202
#Maximum Drawdown:            0.694680
#Calmar Ratio:                4.392622
#Max Drawdown Duration:       168 Days
#Kelly Criterion:             1.937672
#====================================================================================================
