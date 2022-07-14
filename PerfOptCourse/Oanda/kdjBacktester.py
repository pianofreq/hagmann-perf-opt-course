#move into notebook for testing

import pandas as pd
pd.options.display.max_rows = 100
pd.options.display.max_columns = 30
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
import ta
import seaborn as sns
from scipy.optimize import minimize

plt.style.use('seaborn')

class KDJBacktester():
    '''This is a docstring'''
    def __init__(self, filepath, symbol, start, end, tc):
        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.data = None
        self.results = None
        self.parameters = None
        self.get_data()
        self.td_year = (self.data.c.count() / ((self.data.index[-1] - self.data.index[0]).minutes / 365.25*24*60))
        self.best_sol = 0
        
    def __repr__(self):
        return "KDJBacktester: symbol = {}, start = {}, end = {}, ptc = {}".\
        format(self.symbol, self.start, self.end, self.tc)
        
    def get_data(self):
        raw = pd.read_csv(self.filepath, parse_dates=['time'], index_col='time')
        raw = raw.fillna(method = 'ffill')
        raw = raw.loc[self.start:self.end].copy()
        #raw.rename(columns={self.symbol:'price'}, inplace=True)
        raw['returns'] = np.log(raw.c / raw.c.shift(1))
        self.data = raw
        '''
        raw = pd.read_csv(self.filepath, parse_dates=['time'], index_col='time')
        raw = raw[self.symbol].to_frame().fillna(method = 'ffill')
        raw = raw.loc[self.start:self.end].copy()
        raw.rename(columns={self.symbol:'price'}, inplace=True)
        raw['returns'] = np.log(raw.price / raw.price.shift(1))
        self.data = raw
        '''
    
    def upsample(self): #review this method
        return
        data = self.results.copy() #this was self.data which eliminated k and d
        resamp = self.results.copy()
        
        data['position'] = resamp.position.shift(1) #review: not sure why this is being called
        data = data.loc[data.index[0]:].copy()
        data.position = data.position.shift(-1).ffill() #review: not sure why this is being called
        data.dropna(inplace=True)
        self.results = resamp
    
    def prepare_data(self):
        parameters = self.parameters
        #parameters
        #(1,14,3,3,20,20,0,80,80,0,100,200)
        #0 = freq (1,1440)
        #1 = window (2,100)
        #2 = s1 (2,40)
        #3 = s2 (2,40)
        #4 = k long thresh (0,80)
        #5 = d long thresh (0,80)
        #6 = diff long thresh (momentum crossover signal) (-50,50)
        #7 = k short thresh (20,100)
        #8 = d short thresh (20,100)
        #9 = diff short thresh (momentum crossover signal) (-50,50)
        #10 = neutral lower bound (50,200)
        #11 = neutral upper bound (50,200)
        #12 = NATR threshold
        
        data = self.data.copy()
        freq = "{}min".format(int(parameters[0]))
        self.window = parameters[1]
        self.s1 = parameters[2]
        self.s2 = parameters[3]
        #resample() requires an agg method, such as last(), first(), mean(), etc
        #don't resample with daily bars
        resamp = data.resample(freq).last().dropna().iloc[:-1] #the last bar is incomplete, so drop it
        resamp['returns'] = np.log(resamp.c/resamp.c.shift(1)) #take returns on the resampled data
        #resamp['roll_return'] = resamp['returns'].rolling(window).mean()
        #resamp['position'] = -np.sign(resamp['roll_return'])
        ##############
        ###############
        ###############
        
        #resamp["sma"] = resamp.c.rolling(self.sma).mean()
        resamp["k"] = ta.momentum.stochrsi_k(resamp.c, int(parameters[1]), int(parameters[2]), int(parameters[3]), fillna=True)*100
        resamp["d"] = ta.momentum.stochrsi_d(resamp.c, int(parameters[1]), int(parameters[2]), int(parameters[3]), fillna=True)*100
        resamp.dropna(inplace=True)
        resamp['NATR'] = ta.volatility.AverageTrueRange(resamp.h, 
                    resamp.l, resamp.c, window=5).average_true_range()
        resamp = resamp.iloc[14:].copy()

        l_con = ((resamp.NATR > 0.000) & (resamp.k < parameters[4]) & (resamp.d < parameters[5]) & (resamp.d - resamp.k <= parameters[6]))
        s_con = ((resamp.NATR > 0.000) & (resamp.k > parameters[7]) & (resamp.d > parameters[8]) & (resamp.k - resamp.d <= parameters[9]))
        n_con = ((resamp.NATR > 0.000) & (resamp.k+resamp.d>parameters[10])&(resamp.k+resamp.d<parameters[11]))
        resamp["position"] = np.where(l_con, 1, np.nan)
        resamp["position"] = np.where(s_con, -1, resamp["position"])
        resamp["position"] = np.where(n_con, 0, resamp["position"])
        #resamp["position"] = np.where(resamp.distance * resamp.distance.shift(1) < 0, 0, resamp["position"])
        resamp["position"] = resamp.position.ffill().fillna(0)
        #display(resamp.head(500))
        #return
        ##############
        ###############
        ###############
        resamp.dropna(inplace=True)
        self.results = resamp
        return resamp

    def run_backtest(self):
        data = self.results.copy()
        data['strategy'] = data['position'].shift(1) * data['returns']
        
        data['trades'] = data.position.diff().fillna(0).abs() #finds the difference between the rows in 'position'
        data['ctrades'] = data.trades.cumsum()
        data.strategy = data.strategy - data['trades'] * self.tc #subtract trading costs from strategy
        self.results = data
    
    def test_strategy(self, parameters):
        self.parameters = parameters
        #print('trying with these parameters: {}'.format(self.parameters[0:5]))
        #self.parameters = [int(round(item,0)) for item in self.parameters]

        
        #return
        
        self.freq = "{}".format(parameters[0])
        self.prepare_data()
        self.upsample()
        self.run_backtest()
        data = self.results.copy()
        #display(self.results.columns)
        
        data['creturns'] = data['returns'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        #print('^^^ self.results at top of test strat')
        self.results = data
        #print('ran test with these parameters: {}'.format(self.parameters[4:12]))
        #print(-data.cstrategy.iloc[-1])
        if -data.cstrategy.iloc[-1] < self.best_sol:
            self.best_sol = -data.cstrategy.iloc[-1]
            self.best_params = self.parameters
            print(f'{self.start} to {self.end}, new best: {data.cstrategy.iloc[-1]}')
        return -data.cstrategy.iloc[-1]
        #return round(self.calculate_sharpe(data.strategy),6)*-1
          
    def plot_results(self, leverage = False):
        if self.results is None:
            print('run test_strategy() first')
            return
        elif leverage:
            self.results[['creturns', 'cstrategy', 'cstrategy_levered']].plot(logy=False, figsize=(15,8),\
            title = '{} | Frequency: {} | KDJ({}, {}, {}) | TC: {:.6f} | Leverage: {:.2f}'.\
            format(self.symbol, self.parameters[0], self.parameters[1], 
                   self.parameters[2], self.parameters[3], round(self.tc,6), self.leverage))
        else:    
            self.results[['creturns', 'cstrategy']].plot(logy=False, figsize=(15,8),\
            title = '{} | Frequency: {} | KDJ({}, {}, {}) | TC: {:.6f}'.\
            format(self.symbol, self.parameters[0], self.parameters[1],
                   self.parameters[2], self.parameters[3], round(self.tc,6)))
            
    def optimize_strategy(self, window_range, s1_range, s2_range,  metric = "Multiple"):
        
        self.metric = metric
        oParams = self.parameters
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe
        elif metric == "Sortino":
            performance_function = self.calculate_sortino
        elif metric == "Calmar": 
            performance_function = self.calculate_calmar
        elif metric == "Kelly": 
            performance_function = self.calculate_kelly_criterion
        
        
        performance = []
        i = 0
        windows = range(*window_range)
        s1s = range(*s1_range)
        #return np.linspace(*dev_range) #############################################################################
        s2s = range(*s1_range)
        combinations = list(product(windows, s1s, s2s))
        
        for comb in combinations:
            clear_output(wait=True)
            display('Optimization in progress... Iteration {} of {}. Testing: window = {}, s1 = {}, s2 = {}'.\
                    format(i, len(combinations), comb[0], comb[1], comb[2]))
            
            parameters = self.parameters
            parameters[4] = comb[0]
            parameters[5] = comb[0]
            #parameters[6] = 
            parameters[7] = comb[1]
            parameters[8] = comb[1]
            parameters[10] = comb[2]
            parameters[11] = comb[2]+10
            #parameters
            #(1,14,3,3,20,20,0,80,80,0,100,200)
            #0 = freq (1,1440)
            #1 = window (2,100)
            #2 = s1 (2,40)
            #3 = s2 (2,40)
            #4 = k long thresh (0,80)
            #5 = d long thresh (0,80)
            #6 = diff long thresh (momentum crossover signal) (-50,50)
            #7 = k short thresh (20,100)
            #8 = d short thresh (20,100)
            #9 = diff short thresh (momentum crossover signal) (-50,50)
            #10 = neutral lower bound (50,200)
            #11 = neutral upper bound (50,200)
            self.test_strategy(parameters)
            self.upsample() #review: not sure why this is being called
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))
            i += 1
            
        self.results_overview = pd.DataFrame(data = np.array(combinations), columns = ['Window', 'S1', 'S2'])
        self.results_overview['Performance'] = performance
         
        self.find_best_strategy()
            
    
    def find_best_strategy(self):
        self.best = self.results_overview.nlargest(1, "Performance")
        window = self.best.Window.iloc[0]
        s1 = self.best.S1.iloc[0]
        s2 = self.best.S2.iloc[0]
        perf = self.best.Performance.iloc[0]
        clear_output(wait=True) #clear output again after last iteration
        print(100 * "=")
        print('BEST COMBINATION: KDJ({}, {}, {}) | {} = {}'.format(window, s1, s2, self.metric, round(perf,6)))
        print(100 * "-")
        print("\n")
        
        parameters = self.parameters
        parameters[4] = window
        parameters[5] = window
        #parameters[6] = s2
        parameters[7] = s1
        parameters[8] = s1
        parameters[10] = s2
        parameters[11] = s2+10
        
        self.test_strategy(parameters)
        
    def visualize_many(self):
        if self.results is None:
            print('Run test_strategy() first.')
        else:
            matrix = self.results_overview.pivot_table(index='Window', columns='S1', values='Performance', aggfunc='max')
            plt.figure(figsize=(15,8))
            sns.set_theme(font_scale=1.5)
            sns.heatmap(matrix, cmap = 'RdYlGn', robust=True, cbar_kws = {"label":'{}'.format(self.metric)})
            plt.show()

        
        #(1,14,3,3,20,20,0,80,80,0,100,200)
        #0 = freq (1,1440)
        #1 = window (2,100)
        #2 = s1 (2,40)
        #3 = s2 (2,40)
        #4 = k long thresh (0,80)
        #5 = d long thresh (0,80)
        #6 = diff long thresh (momentum crossover signal) (-50,50)
        #7 = k short thresh (20,100)
        #8 = d short thresh (20,100)
        #9 = diff short thresh (momentum crossover signal) (-50,50)
        #10 = neutral lower bound (50,200)
        #11 = neutral upper bound (50,200)
        
    def quick_optimize(self, para, opt_bounds):
        parameters = para
        bnds = opt_bounds
        start_par = para
        #run optimization based on function to be minimized, starting with start parameters
        opts = minimize(fun=self.test_strategy, x0=start_par, method = "Powell" , bounds = bnds, 
        options={'xtol': 0.00001, 'ftol': 0.00001, 'maxiter': None, 
                 'maxfev': None, 'disp': False, 'direc': None, 'return_all': False})
        return opts
        
    def add_sessions(self, visualize = False):
        data = []
        data = self.results.copy()
        
        data['session'] = np.sign(data.trades).cumsum().shift(1).fillna(0)
        
        #cumulative log returns per trading session
        data['logr_session'] = data.groupby('session').strategy.cumsum() 
        
        #cumulative simple returns per trading session
        data['session_compound'] = data.groupby('session').strategy.cumsum().apply(np.exp) - 1
        
        self.results = data
        #print('self.results was changed in add sessions with: ')
        #display(data.columns)
        
        if visualize:
            data['session_compound'].plot(figsize=(12,8))
            plt.show()
        
    def add_stop_loss(self, sl_thresh, report = True):
        
        
        self.sl_thresh = sl_thresh
        
        if self.results is None:
            print('Run test_strategy() first.')
        
        self.add_sessions()
        #print('called add sessions in add_stop_loss')
        
        data = self.results.copy()
        self.results = self.results.groupby('session').apply(self.define_sl_pos)
        self.run_backtest()
        
        data['creturns'] = data['returns'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        
        self.add_sessions(report)
        
        if report:
            self.print_performance()
            
        self.results = data
    
    def add_take_profit(self, tp_thresh, report = True):
        self.tp_thresh = tp_thresh
        
        if self.results is None:
            print('Run test_strategy() first.')
        
        #self.add_sessions()
        #print('called add sessions in add_take_profit')

        self.results = self.results.groupby('sessions').apply(self.define_tp_pos)
        self.run_backtest()
        data = self.results.copy()
        data['creturns'] = data['returns'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        self.add_sessions(report)
        
        if report:
            self.print_performance()
    
    def define_sl_pos(self, group):
        if (group.session_compound <= self.sl_thresh).any():
            start = group[group.session_compound <= self.sl_thresh].index[0]
            #if(len(group)<2):
                #print('here is a group')
                #display(group)
                #return
            stop = group.index[-2]
            group.loc[start:stop, 'position'] = 0
            return group
        else:
            return group
    
    def define_tp_pos(self, group):
        if (group.session_compound >= self.tp_thresh).any():
            start = group[group.session_compound >= self.tp_thresh].index[0]
            stop = group.index[-2]
            group.loc[start:stop, 'position'] = 0
            return group
        else:
            return group
    
    def add_leverage(self, leverage, sl = -0.5, report = True):
        #display(self.results.columns)
        #print('^^self.res from top of add lev')
        self.leverage = leverage
        sl_thresh = sl / leverage #remember, leverage multiplies losses, so you must cut the sl!
        
        self.add_stop_loss(sl_thresh, report = False)
        
        data = self.results.copy()
        #print('self.results in add lev')
        #display(self.results.columns)
        #print('last line in add leverage before artifical return')
        
        data['simple_ret'] = np.exp(data.strategy) - 1 
        data['eff_lev'] = leverage * (1 + data.session_compound) / (1 + data.session_compound * leverage)
        data.eff_lev.fillna(leverage, inplace=True)
        data.loc[data.trades != 0, 'eff_lev'] = leverage
        levered_returns = data.eff_lev.shift() * data.simple_ret
        levered_returns = np.where(levered_returns < -1, -1, levered_returns)
        data['strategy_levered'] = levered_returns
        data['cstrategy_levered'] = data.strategy_levered.add(1).cumprod()
        self.results = data
        #print('self.results was changed in add lev with: ')
        #display(data.columns)
        if report:
            self.print_performance(leverage = True)
        
    
    def print_performance(self, leverage = False):
        data = self.results.copy()
        
        if leverage:
            to_analyze = np.log(data.strategy_levered.add(1))
        else:
            to_analyze = data.strategy
        
        bh_multiple =               round(self.calculate_multiple(data.returns),6)
        strategy_multiple =         round(self.calculate_multiple(to_analyze),6)
        strategy_cagr =             round(self.calculate_cagr(to_analyze),6)
        outperf =                   round(strategy_multiple - bh_multiple, 6)
        strategy_ann_mean =         round(self.calculate_annulized_mean(to_analyze),6)
        strategy_ann_std =          round(self.calculate_annulized_std(to_analyze),6)
        strategy_sharpe =           round(self.calculate_sharpe(to_analyze),6)
        strategy_sortino =          round(self.calculate_sortino(to_analyze),6)
        strategy_max_dd =           round(self.calculate_max_drawdown(to_analyze),6)
        strategy_calmar =           round(self.calculate_calmar(to_analyze),6)
        strategy_max_dd_dur =       round(self.calculate_max_dd_duration(to_analyze),6)
        strategy_kelly =            round(self.calculate_kelly_criterion(to_analyze),6)
        strategy_ctrades =          round(self.results.ctrades[-1])
        
        print(100 * "=")
        title="OPTIMIZED KDJ STRATEGY | INSTRUMENT = {} | FREQ = {} | KDJ({}, {}, {})"
        print(title.format(self.symbol,\
                self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]))
        print(100 * "-")
        print("\n")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {:,.6f}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {:,.6f}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {:,.6f}".format(outperf))
        print("\n")
        print("CAGR:                        {:,.6f}".format(strategy_cagr))
        print("Annualized Mean:             {:,.6f}".format(strategy_ann_mean))
        print("Annualized Std:              {:,.6f}".format(strategy_ann_std))
        print("Sharpe Ratio:                {:,.6f}".format(strategy_sharpe))
        print("Sortino Ratio:               {:,.6f}".format(strategy_sortino))
        print("Maximum Drawdown:            {:,.6f}".format(strategy_max_dd))
        print("Calmar Ratio:                {:,.6f}".format(strategy_calmar))
        print("Max Drawdown Duration:       {} Days".format(strategy_max_dd_dur))
        print("Total Trades:                {} Trades".format(strategy_ctrades))
        print("Kelly Criterion:             {:,.6f}".format(strategy_kelly))
        
        print(100 * "=")
        
    
    def calculate_multiple(self, series):
        #return series.iloc[-1]/series.iloc[0] #this returns multiple if the series is PRICE data
        return np.exp(series.sum()) #returns muliple if the series is log returns (does work with nan in 0 pos)
        #return series.cumsum().apply(np.exp).iloc[-1]
    
    def calculate_cagr(self, series):
        return np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).minutes / 365.25*24*60)) - 1
    
    def calculate_annulized_mean(self, series):
        return series.mean() * self.td_year
    
    def calculate_annulized_std(self, series):
        return series.std() * np.sqrt(self.td_year)
    
    def calculate_sharpe(self, series):
        if series.std() == 0:
            return np.nan
        else:
            return series.mean() / series.std() * np.sqrt(self.td_year)
    
    def calculate_sortino(self, series):
        excess_returns = (series - 0)
        downside_deviation = np.sqrt(np.mean(np.where(excess_returns < 0, excess_returns, 0)**2))
        if downside_deviation == 0:
            return np.nan
        else:
            sortino = (series.mean() - 0) / downside_deviation * np.sqrt(self.td_year)
            return sortino 
    
    def calculate_max_drawdown(self, series):
        creturns = series.cumsum().apply(np.exp)
        cummax = creturns.cummax()
        drawdown = (cummax - creturns)/cummax
        max_dd = drawdown.max()
        return max_dd
    
    def calculate_calmar(self, series):
        max_dd = self.calculate_max_drawdown(series)
        if max_dd == 0:
            return np.nan
        else:
            cagr = self.calculate_cagr(series)
            calmar = cagr / max_dd
            return calmar
    
    def calculate_max_dd_duration(self, series):
        creturns = series.cumsum().apply(np.exp)
        cummax = creturns.cummax()
        drawdown = (cummax - creturns)/cummax
    
        begin = drawdown[drawdown == 0].index
        end = begin[1:]
        end = end.append(pd.DatetimeIndex([drawdown.index[-1]]))
        periods = end - begin
        max_ddd = periods.max()
        return max_ddd.minutes/60/24
    
    def calculate_kelly_criterion(self, series):
        series = np.exp(series) - 1
        if series.var() == 0:
            return np.nan
        else:
            return series.mean() / series.var()