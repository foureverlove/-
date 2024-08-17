import pandas as pd
import numpy as np
import warnings

def get_basic_data():
    df = pd.read_csv('factor_model_fillna.csv')
    return df

def get_target_list(df,trade_date):
    df = df[df["trade_date"]== trade_date]
    x1 = df.close < 100
    x2 = df.turnover_rate > 15
    x3 = df.circ_mv>500000
    x4 = df.circ_mv < 3000000
    x = x1&x2&x3&x4
    stock_list = df[x].ts_code.values
    return stock_list


#获取股票收益率

def risk_budget_objective(weights,cov,x0):
    x0 = np.array(x0) #当x0等权重时为风险平价
    weights = np.array(weights) #weights为一维数组
    sigma = np.sqrt(np.dot(weights, np.dot(cov, weights))) #获取组合标准差  
    MRC = np.dot(cov,weights)/sigma  
    TRC = weights * MRC
    #print(TRC)
    risk_target = sigma * x0
    #print(risk_target)
    delta_TRC = [sum((i - TRC)**2) for i in risk_target]
    return sum(delta_TRC)

# Constraint: 
def total_weight_constraint(x):
    return np.sum(x) - 1.0

#def constraint1(weights,cov,x0):
    





