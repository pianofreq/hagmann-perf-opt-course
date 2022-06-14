#!/usr/bin/env python3

#run in termimal by typing: nohup python3 trader-rtm-EURAUD.py & (include &)
#https://coduber.com/run-python-script-constantly-in-background/

import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time
import os

class ConTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, SMA, dev, weight = 0): 
        #set to a number between 0 and 1 for weight
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.weight = weight
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
            df
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
                break
                
    def start_trading(self, days, max_attempts = 5, wait = 20, wait_increase = 0): # Error Handling
        attempt = 0
        success = False
        while True:
            try:
                self.get_most_recent(days)
                self.stream_data(self.instrument)
            except Exception as e:
                print(e, end = " | ")
            else:
                success = True
                break    
            finally:
                attempt +=1
                print("Attempt: {}".format(attempt), end = '\n')
                if success == False:
                    if attempt >= max_attempts:
                        print("max_attempts reached!")
                        try: # try to terminate session
                            time.sleep(wait)
                            self.terminate_session(cause = "Unexpected Session Stop (too many errors).")
                        except Exception as e:
                            print(e, end = " | ")
                            print("Could not terminate session properly!")
                        finally: 
                            break
                    else: # try again
                        time.sleep(wait)
                        wait += wait_increase
                        self.tick_data = pd.DataFrame()
        
                
    def on_success(self, time, bid, ask):
        
        print(self.ticks, end = " ", flush = True)
        
        recent_tick = pd.to_datetime(time)
        
        # define stop
        if recent_tick.time() >= pd.to_datetime("22:30").time():
            self.stop_stream = True
        
        df = pd.DataFrame({self.instrument:(ask + bid)/2, 'spread':(bid-ask)},
                          index = [recent_tick])
        self.tick_data = self.tick_data.append(df)
        #print('recent t: {}\n'.format(recent_tick),\
        #      'last bar: {}\n'.format(self.last_bar), 'bar l: {}\n'.format(self.bar_length), 'logic: {}'.format(recent_tick - self.last_bar > self.bar_length))
        if recent_tick - self.last_bar > self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()
    
    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                                  label="right").last().ffill().iloc[:-1])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
        #print(self.raw_data)
    
    def define_strategy(self): # "strategy-specific"
        data = self.raw_data.copy()
        #data
        #print('###################that was raw_data####################')
        #self.raw_data

        #******************** define your strategy here ************************
        ''' Backtests the Bollinger Bands-based trading strategy.
        '''
        data['returns'] = np.log(data[self.instrument] / data[self.instrument].shift(1))
        data["SMA"] = data[self.instrument].rolling(self.SMA).mean()

        data["Lower"] = data["SMA"] - data[self.instrument].rolling(self.SMA).std() * self.dev
        data["Upper"] = data["SMA"] + data[self.instrument].rolling(self.SMA).std() * self.dev
        data["distance"] = data[self.instrument] - data.SMA
        data["position"] = np.where(data[self.instrument] < data.Lower, 1, np.nan)
        data["position"] = np.where(data[self.instrument] > data.Upper, -1, data["position"])
        data["position"] = np.where(data.distance * data.distance.shift(1) < 0, 0, data["position"])
        data["position"] = data.position.ffill().fillna(0)
        data["strategy"] = data.position.shift(1) * data["returns"]
        data.dropna(inplace = True)

        
        # determine the number of trades in each bar
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction/trading costs from pre-cost return
        data.strategy = data.strategy - data.trades * data.spread
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        data['bal'] = round(data['cstrategy'] * float(trader.get_account_summary()['balance']),2)
        self.data = data
        
        #does not apply
        #perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        #outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
        
        #for debugging strategy code
        os.system('clear')
        print('\n')
        print(data.tail(10))
        print('###################that was data####################')
        return        
        #return round(perf, 6), round(outperf, 6)
        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.calc_units(), suppress = True, ret = True)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.calc_units() * 2, suppress = True, ret = True) 
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:                
                order = self.create_order(self.instrument, -self.calc_units(), suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.calc_units() * 2, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        elif self.data["position"].iloc[-1] == 0:
            if self.position == -1:
                order = self.create_order(self.instrument, self.calc_units(), suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.calc_units(), suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0
    
    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")  
        
    def terminate_session(self, cause):
        self.stop_stream = True
        if self.position != 0:
            close_order = self.create_order(self.instrument, units = -self.position * self.units,
                                            suppress = True, ret = True) 
            self.report_trade(close_order, "GOING NEUTRAL")
            self.position = 0
        print(cause, end = " | ")
        
    def calc_units(self):
        return 200000
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                self.bal = round(float(self.get_account_summary()['balance']),2) #get account balance
                self.lev = 19.62 #IMPROVE THIS -- make leverage dynamic from PerfOpt. static leverage from half kelly
                self.units = round(self.bal*self.lev*self.weight,0)
                return self.units
            
            elif self.position == -1:
                #get existing position units
                self.units = float(self.get_positions()[0]['long']['units'])\
                +float(self.get_positions()[0]['short']['units'])
                return self.units
            
        elif self.data["position"].iloc[-1] == -1:
            if self.position == 0:
                self.bal = round(float(self.get_account_summary()['balance']),2) #get account balance
                self.lev = 19.62 #IMPROVE THIS -- make leverage dynamic from PerfOpt. static leverage from half kelly
                self.units = round(self.bal*self.lev*self.weight,0)
                return self.units
            
            elif self.position == 1:
                #get existing position units
                self.units = float(self.get_positions()[0]['long']['units'])\
                +float(self.get_positions()[0]['short']['units'])
                return self.units
            
        elif self.data["position"].iloc[-1] == 0:
            if self.position == -1:
                #get existing position units
                self.units = float(self.get_positions()[0]['long']['units'])\
                +float(self.get_positions()[0]['short']['units'])
                return self.units
            
            elif self.position == 1:
                #get existing position units
                self.units = float(self.get_positions()[0]['long']['units'])\
                +float(self.get_positions()[0]['short']['units'])
                return self.units  
        
if __name__ == "__main__":
    config_file = 'oanda.cfg'
    trader = ConTrader(config_file, "AUD_CAD", "15min", SMA = 20, dev = 1.8947368, weight = 1)
    trader.start_trading(days = 5, max_attempts =  3, wait = 20, wait_increase = 0)
    
#====================================================================================================
#BEST COMBINATION: FREQUENCY = 361.0-MINUTE CHART | SMA = 49.0 | DEV = 1.7058823529411764 | Multiple = 1.280781
#----------------------------------------------------------------------------------------------------        

#====================================================================================================
#SIMPLE CONTRARIAN STRATEGY | INSTRUMENT = EURUSD | FREQ = 361.0 | SMA = 49 | DEV = 1.7058823529411764
#----------------------------------------------------------------------------------------------------


#PERFORMANCE MEASURES:


#Multiple (Strategy):         195.390982
#Multiple (Buy-and-Hold):     0.896314
#--------------------------------------
#Out-/Underperformance:       194.494668


#CAGR:                        14.930564
#Annualized Mean:             2.771689
#Annualized Std:              1.258116
#Sharpe Ratio:                2.203047
#Sortino Ratio:               3.144389
#Maximum Drawdown:            0.612140
#Calmar Ratio:                24.390764
#Max Drawdown Duration:       197 Days
#Kelly Criterion:             2.251164
#====================================================================================================