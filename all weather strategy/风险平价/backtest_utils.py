import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize
from datetime import datetime
from tqdm.auto import tqdm
import copy
from cvxopt import matrix, solvers
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numba import jit
from matplotlib.font_manager import FontProperties
import math
import random

#加杠杆使波动率低于目标波动率的资产到目标波动率
def calculate_vol(data,target_vol,trade_date,vol_day): 
    leverage = {}
    data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    codes = sorted(set(data['ts_code']))
    dates = trade_date
    data_date = pd.to_datetime(data['trade_date']) 
    
    for code in tqdm(codes):
        tep_data = data.loc[data['ts_code'] == code]
        for date in dates:
            daily_returns = []
            one_year_ago = str(pd.to_datetime(date) - timedelta(days= vol_day))
            mask = (tep_data['trade_date'] >= one_year_ago) & (tep_data['trade_date'] < date)
            tep_data = tep_data.sort_values(by = 'trade_date',ascending = True)
            daily_returns = tep_data.loc[mask, "pct_chg"].fillna(0)
            if len(daily_returns) > 1: 
                vol = daily_returns.std()* math.sqrt(252)
            if vol < target_vol:
                if target_vol/vol <=10:
                    leverage[(code, date)] = target_vol / vol
                    data.loc[(data['ts_code'] == code) & (data_date == date), 'pct_chg'] *= float(target_vol / vol)
                else:
                    leverage[(code, date)] = 10
                    data.loc[(data['ts_code'] == code) & (data_date == date), 'pct_chg'] *= 10
            else:
                leverage[(code, date)] = 1
                
    return data,leverage

# 计算资产与其他资产的协方差
def calculate_cov(data,now_date,vol_day):
    
    ret = pd.DataFrame()
    now_date = datetime.strptime(str(now_date), '%Y%m%d')
    codes = list(set(data['ts_code']))
    data_stock = data['ts_code']
    data_date = pd.to_datetime(data['trade_date'])
        
    for code in codes:
        one_year_ago = str(pd.to_datetime(now_date) - timedelta(days=vol_day))
        mask = (data_stock == code) & (data_date < now_date) & (data_date >= one_year_ago)
        ret_ = data.loc[mask]
        ret_ = ret_.sort_values(by = ['trade_date'],ascending = True)
        ret_ = ret_["pct_chg"].reset_index(drop=True)
        
        if not ret_.empty:
            if ret_.isna().any():
                ret_ = ret_.fillna(value=ret_.mean())
            ret[code] = ret_

    R_cov = ret.cov()
    
    cov = np.array(R_cov)
    return cov,R_cov.columns


def set_random_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    
@jit(nopython=True)
def risk_budget_objective(weights, cov, x0):
    weights = np.asarray(weights)
    x0 = np.asarray(x0) 

    sigma = np.sqrt(np.dot(weights, np.dot(cov, weights))) # 获取组合标准差  
    MRC = np.dot(cov, weights) / sigma  
    TRC = weights * MRC
    risk_target = sigma * x0
    delta_TRC = np.sum((risk_target - TRC) ** 2)
    return delta_TRC


def total_weight_constraint(x):
    return np.sum(x) - 1.0

def calculate_weight(cov, x0):
    # Bounds: 无卖空
    bounds = tuple((0, 1) for _ in x0)
    
    # 约束
    cons = ({'type': 'eq', 'fun': total_weight_constraint})
    
    # 优化
    options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-12}
    solution = minimize(risk_budget_objective, x0, args=(cov, x0), bounds=bounds, constraints=cons, method='SLSQP', options=options)

    # 计算权重结果
    final_weights = solution.x
    return final_weights

#计算支持性数据
def supporting_data(data, trade_date, stocks, vol_day):
    # 预先计算协方差矩阵，避免在循环中重复计算
    covs = {date: calculate_cov(data, date, vol_day)[0] for date in trade_date}
    x0 = np.ones(len(stocks)) / len(stocks)  # 风险平价

    stock_weights = {stock: [] for stock in stocks}
    date_weights = {date: [] for date in trade_date}
    date_stocks = {date: [] for date in trade_date}

    for date in tqdm(trade_date):
        cov = covs[date]
        final_weights = calculate_weight(cov, x0)  

        #if len(stocks) == len(final_weights):
        for i, stock in enumerate(stocks):
            stock_weights[stock].append(final_weights[i])
            date_weights[date].append(final_weights[i])
            date_stocks[date].append(stock)
        #else:
            #print(f"股票列表和权重列表长度不匹配在日期 {date}")

    return stock_weights, date_weights, date_stocks

# all weather四个环境资产权重叠加
def four_quarters(data,trade_date,stocks_dict,vol_day):
    results = {}
    # 循环处理每个股票组和杠杆组
    for i in range(1, 5):
        stocks = stocks_dict[f'stocks_{i}']
        data_filtered = data.loc[data['ts_code'].isin(stocks)]
        stock_weights, _, _ = supporting_data(data_filtered, trade_date, stocks,vol_day)
        results[f'stock_weights_{i}'] = stock_weights
    
    return results



    
def calculate_daily_returns(leverage,df, date_stocks,date_weights,initial_capital, fee_rate,deviate_rate,bond_rate,trade_date):
    #date_stocks: 每天买入的股票 type字典
    #df 每天股市各股票基本信息

    net_value = initial_capital
    net_values = []  # 初始净值
    daily_returns = []
    expected_holdings = []
    actual_holdings = []
    position_change = []
    holdings_diffs = []
    stock_earnings = []
    stock_earning = []
    
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    fixed_trade_date =pd.to_datetime(trade_date[0])
    df = df.set_index(["ts_code","trade_date"])
    df = df["pct_chg"]/100
    position = 1
    
    for date, stock_list in tqdm(date_stocks.items(), desc="计算日收益"):
            free_rate = float((bond_rate[bond_rate.index == date].iloc[0] / 25200))
            return_list = []
            #判断持仓是否为空
            if len(actual_holdings) == 0:
                actual_holdings = date_weights[date]

            #利用风险平价计算的持仓
            expected_holdings = date_weights[date]

            if position == 1:
                leverage_list = []

            #持仓的日收益率    
            for stock in stock_list: 
                
                if position == 1:
                    if (stock, date) in df.index:

                        leverage_list.append(leverage[(stock,date)])
                        return_ = float(df.loc[(stock, date)]- free_rate)
                        print(return_)
                        return_list.append(return_)

                    else: 
                        leverage_list.append(1)
                        return_ = 0 - free_rate
                        return_list.append(return_)


                elif position == 0:
                    if (stock, date) in df.index:
                        return_ = float(df.loc[(stock, date)]- free_rate)
                        return_list.append(return_)


                    else:
                        return_ = 0 - free_rate
                        return_list.append(return_)


            multiplied_result =np.array(leverage_list) * np.array(return_list)
            free_rate_array = np.full_like(multiplied_result, free_rate)
            return_list = multiplied_result + free_rate_array
            return_list = return_list.tolist()
            
            if stock_earning is None or len(stock_earning) == 0:
                stock_earning = np.zeros(len(return_list))

            stock_earning += net_value * np.array(return_list) * np.array(actual_holdings)
            stock_earnings.append(stock_earning.copy())
            
            #判断是否调整持仓 0不换 1换
            for i in range(len(expected_holdings)):
                diff_holding = actual_holdings[i]-expected_holdings[i]
                if abs(diff_holding)<= deviate_rate and pd.to_datetime(date) <= fixed_trade_date :

                    position = 0

                else:
                    position = 1
                    fixed_trade_date = pd.to_datetime(date) + timedelta(days = 90)
                    break

            holdings_diff = [actual - expected for actual,expected in zip(actual_holdings,expected_holdings)]
            holdings_diffs.append(holdings_diff)
            
            
            if position == 1:
                #print(f"{date}需要调整持仓")
                position_change.append(date)
                #计算持仓比例变化与交易费用


                abs_list = list(map(abs, holdings_diff))
                transaction_cost = sum(abs_list)* net_value * fee_rate
                net_value -= transaction_cost
                actual_holdings = copy.deepcopy(expected_holdings)


                adjusted_returns = [1 + returns for returns in return_list]                            
                # 计算新的净值
                new_values = [net_value * actual * adjusted_return for actual,adjusted_return in zip(actual_holdings,adjusted_returns)]
                #print(f'改日净值为：{np.sum(new_values) - net_value}')
                returns = np.sum(new_values)/net_value - 1
                daily_returns.append(float(returns))
                net_value = np.sum(new_values)
                actual_holdings = new_values/np.sum(new_values) 
                net_values.append(net_value)

            elif position == 0 :
                #print(f"{date}不需要调整持仓")

                #计算
                adjusted_returns = [1 + returns for returns in return_list]     
                # 计算新的净值
                new_values = [net_value * actual * adjusted_return for actual,adjusted_return in zip(actual_holdings,adjusted_returns)]
                returns = np.sum(new_values)/net_value - 1
                #print(returns*100)
                daily_returns.append(float(returns))
                net_value = np.sum(new_values)
                actual_holdings = new_values/np.sum(new_values)
                net_values.append(net_value)

            if len(leverage_list) != len(stock_list):
                position = 1
                
    return daily_returns,net_values,position_change,holdings_diffs,stock_earnings,stock_list

def plot_cumulative_returns(daily_returns,trade_date):
    # 累计收益率，即每一天都加上前一天的收益率
    cumulative_returns = (1 + pd.Series(daily_returns)).cumprod() - 1
    trade_date = pd.to_datetime(trade_date, format = '%Y%m%d')
    plt.figure(figsize=(12, 6))
    plt.plot(trade_date,cumulative_returns,label = "cumulative_returns")
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative returns_2.png', format='png', dpi=300)
    plt.show()

#基本指标计算
def calculate_performance_metrics(daily_returns):
    daily_returns = pd.Series(daily_returns)
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    win_rate = (daily_returns > 0).mean()

    sharpe_ratio = daily_returns.mean() / daily_returns.std()* np.sqrt(252)
    
    maximum = 0
    drawdown = []
    for i in range(len(cumulative_returns)):
        if cumulative_returns[i]> maximum:
            maximum = cumulative_returns[i]
            
        elif cumulative_returns[i]<= maximum:
            drawdown.append(maximum - cumulative_returns[i])
    max_drawdown = max(drawdown)
    
    return  win_rate, sharpe_ratio, max_drawdown

def plot_etf_returns(data, test_dt):

    data = data.sort_values(by=['ts_code', 'trade_date'], ascending=True)
    data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')

    data = data[data['trade_date'] >= test_dt]

    stocks = data['ts_code'].unique()
    all_trade_dates = pd.to_datetime(data['trade_date'].unique())

    full_data = pd.DataFrame()

    for stock in stocks:
        stock_data = data[data['ts_code'] == stock].set_index('trade_date')
        stock_data = stock_data.reindex(all_trade_dates, method='ffill')
        stock_data = stock_data.reset_index() [['ts_code', 'trade_date', 'pct_chg']]
        full_data = pd.concat([full_data, stock_data], ignore_index=True)

    full_data = full_data.sort_values(by=['ts_code', 'trade_date'], ascending=True)

    full_data['pct_chg'] = full_data['pct_chg'] / 100
    plt.figure(figsize=(14, 7))

    codes = full_data['ts_code'].unique()
    for code in codes:
        daily_returns = full_data[full_data['ts_code'] == code]['pct_chg']
        cumulative_returns = (1 + pd.Series(daily_returns)).cumprod() - 1
        plt.plot(all_trade_dates, cumulative_returns, label=code)

    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.legend()
    plt.show()

#业绩归因
def cal_asset(daily_returns,stock_earnings,stock_list,trade_date):
    cumulative_returns = (1 + pd.Series(daily_returns)).cumprod() - 1
    earning_percentages = []
    
    for i,earning in enumerate(stock_earnings):
        total_earning = sum(earning)
        earning_percentage = earning/total_earning
        earning_percentages.append(earning_percentage * cumulative_returns[i])
                                   
    stock_daily_returns = {stock:[] for stock in stock_list}
    
    for array in earning_percentages:
        for i,stock in enumerate(stock_list):
            stock_daily_returns[stock].append(list(array)[i])
            
    plt.figure(figsize=(12, 6))
    
    portfolio_returns = cumulative_returns
    plt.plot(trade_date, portfolio_returns, label=" Portfolio Returns")
    
    for stock,stock_daily_return in stock_daily_returns.items():
        cumulative_returns_array = np.array(cumulative_returns)
        cumulative_returns = np.clip(cumulative_returns_array, -1, 3)
        plt.plot(trade_date, stock_daily_return, label=stock)
    plt.title('Asset Attribute')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# smart_beta    
def cal_smart_beta(daily_returns,stock_earnings,stock_list,trade_date,stock_weights_sum,results):
    #获得净值时间序列
    cumulative_returns = (1 + pd.Series(daily_returns)).cumprod() - 1
    stock_values = {stock:[] for stock in stock_list}
    for stock_earning in stock_earnings:
        for i,stock in enumerate(stock_list):
            stock_values[stock].append(stock_earning[i])

    section_values = {}
    section_name = ['高通胀','高增长','低增长','低通胀']
    for i,weights in enumerate(results.values()):
        for stock,weight in weights.items():
            value = np.array(weight)/stock_weights_sum[stock]* np.array(stock_values[stock])
            if section_name[i] not in section_values.keys():
                section_values[section_name[i]] = value
            else:
                section_values[section_name[i]] += value
    
    total_value = sum(section_values.values())
    
    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(trade_date,cumulative_returns,label = "portfolio_returns")
    
    section_cum = []
    
    for section_name,value in section_values.items():
        section_series = value / total_value * cumulative_returns
        print(list(section_series)[-1])
        section_cum.append(section_series) 
        section_series = section_series.clip(lower=-0.5, upper=2.5)
        
        plt.plot(trade_date, section_series, label=section_name) 
        
    section_cum.append(cumulative_returns)    
    plt.title('Section Attribute')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()
    return section_cum