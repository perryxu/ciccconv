import pandas as pd
from sklearn.linear_model import LinearRegression

# 筛选符合特定交易量要求的券列表
def selByAmt(obj, date, other=None, noCrazy=True):
    # 选出当天有成交的券（按成交额）；选出当天成交金额大于流通市值的券（换手率大于100%）；乘号是boolean的并集
    # list(iterator)，list是可以接收iterator的，list(range(10))
    t = obj.DB['Amt'].loc[date] > 0
    t *= obj.DB['Amt'].loc[date] < (obj.DB['Close'].loc[date] * obj.DB['Outstanding'].loc[date] / 100.0)

    if other:
        for k, v in other.iteritems():
            t *= obj.DB[k].loc[date].between(v[0], v[1])
    # 技巧：t是true、false序列，t[t]可以筛选出只含true的短序列（可以取index）
    codes = list(t[t].index)
    return codes

def selByAmtPq(obj, start, end):
    # 选出指定时间区间内，有过交易的券编号
    t = obj.DB['Amt'].loc[start:end].sum() > 0
    codes = list(t[t].index)
    return codes

def getCBReturn(date, codes=None, obj=None):
    # obj为None时，注意语法作用；codes为None时，注意语法用法。
    if not obj:
        obj = cb.cb_data()
    if not codes:
        codes = selByAmt(obj, date)
    # 得到index中等于date的次序值。
    loc = obj.DB['Amt'].index.get_loc(date)
    # 得到指定日期的较前一日收益率。
    return 100.0 * (obj.DB['Close'][codes].iloc[loc] / obj.DB['Close'][codes].iloc[loc-1] - 1.0)

# 处理行业因子
def getUnderlyingCodeTable(codes):
    '''
    输入转债代码list
    返回正股代码list
    '''
    sql = """select a.s_info_windcode cbCode,b.s_info_windcode underlyingCode
    from winddf.ccbondissuance a,winddf.asharedescription b
    where a.s_info_compcode = b.s_info_compcode and
    length(a.s_info_windcode) = 9 and
    substr(a.s_info_windcode,8,2) in ('SZ','SH') and
    substr(a.s_info_windcode,1,3) not in ('137','117')"""
    con = login(1) # 为我们的万得数据链接对象
    ret = pd.read_sql(sql, con, index_col='CBCODE')
    return  ret.loc[codes]

def cbInd(codes):
    dfUd = getUnderlyingCodeTable(codes)
    sql = '''select a.s_info_windcode as udCode,
    b.industriesname indName
    from 
    winddf.ashareindustriesclasscitics a,
    winddf.ashareindustriescode b
    where substr(a.citics_ind_code,1,4) = substr(b.industriescode,1,4) and
    b.levelnum = '2' and
    a.cur_sign = '1' and
    a.s_info_windcode in ({_codes})
    '''.format(_codes = rsJoin(list(set(dfUd['UNDERLYINGCODE']))))
    con = login(1) # 为我们的万得数据链接对象
    dfInd = pd.read_sql(sql, con, index_col='UDCODE')
    # 可能有的券行业数据None。
    dfUd['Ind'] = dfUd['UNDERLYINGCODE'].apply(lambda x: dfInd.loc[x, 'INDNAME'] if x in dfInd.index else None)
    return dfUd['Ind']

# 构成行业矩阵，0/1阵
# 关于潜在的过拟合：如果因子数大于样本数的一半，可能发生过拟合；最好在1/4以下
def factorInd(codes, cbInd=None):
    if not cbInd:
        cbInd = pd.DataFrame({'ind':_cbInd(codes)})
    cbInd = pd.merge(cbInd, indCls, left_on='ind', right_index=True)
    dfRet = pd.DataFrame(index=codes, columns=set(cbInd['ind']))
    # 可以用multilayer index和unstack来优化
    for c in dfRet.columns:
        tempCodes = cbInd.loc[cbInd['ind'].apply(lambda x:x.encode('gbk')) == c.encode('gbk')].index
        dfRet.loc[tempCodes, c] = 1.0
    return dfRet.fillna(0)

# 计算流通市值因子矩阵
def factorSize_cb_outstanding(codes, start, end, obj=None):
    if not obj:
        obj = cb.cb_data()
    ost_mv = obj.DB['Close'].loc[start:end, codes] * obj.DB['Outstanding'].loc[start:end, codes] / 100.0
    return ost_mv

# 计算rank因子处理，div是矢量化计算的除法，rank的多重参数（min/max/average;pct）
def rankCV(df):
    rk = df.rank(axis=1, pct=True)
    return (rk - 0.5).div(rk.std(axis=1), axis='rows')

# 多元回归
def oneFactorReg(start, end, dfFactor, factorName='ToBeTest', dfFctInd=None, obj=None):
    if not obj:
        obj = cb.cb_data()
    if not dfFctInd:
        codes = selByAmtPq(obj, start, end)
        codes = list(set(codes).intersection(list(dfFactor.columns)))
        dfFctInd = factorInd(codes)

    arrDates = list(obj.DB['Amt'].loc[start:end].index)[1:]
    lr = LinearRegression(fit_intercept=True)
    # 事先规定一个column和index确定的空frame
    dfRet = pd.DataFrame(index=arrDates, columns=['One'] + list(dfFctInd.columns) + [factorName, 't', 'score'])
    # 市值因子矩阵
    dfCBMV = factorSize_cb_outstanding(codes, start, end, obj)

    for date in arrDates:
        print(date)

        tCodes = selByAmt(obj, date)
        # Y因变量：每日return
        srsReturn = getCBReturn(date, tCodes, obj)
        # dfX首先填入行业矩阵，横向行业，纵向券码。
        dfX = pd.DataFrame(index=tCodes)
        dfX[list(dfFctInd.columns)] = dfFctInd
        # dfX放入待检验单因子。
        dfX[factorName] = dfFactor.loc[date]
        dfX.dropna(inplace=True)
        idx = dfX.index
        # 权重按流动市值的平方根加总。dfCBMV横向是券号，纵向是日期。
        arrW = pd.np.sqrt(dfCBMV.loc[date, idx])
        arrW /= arrW.sum()
        # fit(self, X, y, sample_weight=None)
        lr.fit(dfX.loc[:, :], srsReturn[idx], arrW)

        dfRet.loc[date, list(dfFctInd.columns) + [factorName]] = lr.coef_
        dfRet.loc[date, 'One'] = lr.intercept_
        dfRet.loc[date, 't'] = t_test(lr, dfX.loc[:, :], srsReturn[idx])

        dfRet.loc[date, 'score'] = lr.score(dfX.loc[:, :], srsReturn[idx], arrW)

    print(pd.np.abs(dfRet['t']).mean())
    print(pd.np.abs(dfRet['score']).mean())
    return dfRet

# t检验
def t_test(lr, x, y):
    n = len(x) * 1.0
    predY = lr.predict(x)
    e2 = sum((predY - y) ** 2)
    varX = pd.np.var(x) * n
    t = lr.coef_ * pd.np.sqrt(varX) / pd.np.sqrt(e2 / n)
    return t[-1]

