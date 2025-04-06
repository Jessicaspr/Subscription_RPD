import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import curve_fit

def log_power_function(x, a, b):
    """定义对数函数: y = log(a * x^b)，用于拟合留存率"""
    return np.log(a * np.power(x, b))

def fit_revenue_parameters(days, retention_rates):
    """拟合留存率衰减参数"""
    # 检查是否所有留存率都为0
    if np.all(retention_rates == 0):
        return 0, 0  # 返回特殊值表示全为0的情况
    
    log_retention_rates = np.log(retention_rates)
    popt, _ = curve_fit(log_power_function, days, log_retention_rates)
    return popt

class RevenueCalculator:
    def __init__(self):
        # 定义不同周期的观测点（使用周期数而不是天数）
        self.week_points = np.array([1, 3, 7])  # 1周、3周、7周
        self.month_points = np.array([1, 3, 7])  # 1月、3月、7月
        self.quarter_points = np.array([1, 3, 4])  # 1季、3季、4季
        self.year_points = np.array([1, 2, 3])  # 1年、2年、3年

    def fit_period_parameters(self, retention_rates, period_type):
        """根据不同周期拟合参数
        retention_rates: 对应周期点的续订率
        period_type: 'week', 'month', 'quarter', 或 'year'
        """
        if period_type == 'week':
            periods = self.week_points
        elif period_type == 'month':
            periods = self.month_points
        elif period_type == 'quarter':
            periods = self.quarter_points
        else:  # year
            periods = self.year_points
        
        return fit_revenue_parameters(periods, np.array(retention_rates))

    def calculate_retention_rate(self, days, a, b, period_type):
        """计算特定天数对应的留存率
        days: 距离激活的天数
        period_type: 'week', 'month', 'quarter', 或 'year'
        """
        # 特殊情况处理：如果a和b都为0，表示该方案不使用
        if a == 0 and b == 0:
            return 0
        
        # 将天数转换为对应的周期数
        if period_type == 'week':
            periods = days / 7
        elif period_type == 'month':
            periods = days / 30
        elif period_type == 'quarter':
            periods = days / 90
        else:  # year
            periods = days / 365
        
        return np.exp(log_power_function(periods, a, b))

    def create_revenue_df(self, active_users, start_date, end_date, yearly_params):
        """创建收入DataFrame"""
        dates = pd.date_range(start=start_date, end=end_date)
        
        # 为每年每种付费周期拟合参数
        year_params = {}
        for year, params in yearly_params.items():
            year_params[year] = {}
            for period in ['week', 'month', 'quarter', 'year']:
                retention_rates, revenue_prop = params[period]
                a, b = self.fit_period_parameters(retention_rates, period)
                year_params[year][period] = (a, b, revenue_prop)
        
        data = {'激活人数': [active_users] * len(dates)}
        base_revenue = 1000  # 总体基础收入
        
        max_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
        
        # 为每一天创建收入列
        for day in range(max_days):
            col_name = f'day{day}收入'
            revenues = []
            
            for idx, date in enumerate(dates):
                if idx + day < len(dates):
                    activation_date = date
                    revenue_date = date + pd.Timedelta(days=day)
                    year = activation_date.year
                    
                    if day == 0:
                        # day0收入直接按收入占比分配
                        revenues.append(base_revenue)
                    else:
                        total_revenue = 0
                        # 计算每种付费周期的收入
                        for period, (period_days, divisor) in {
                            'week': (7, 7),
                            'month': (30, 30),
                            'quarter': (90, 90),
                            'year': (365, 365)
                        }.items():
                            # 只有在当天是对应周期的付费日时才计算收入
                            if day % period_days == 0:
                                a, b, revenue_prop = year_params[year][period]
                                retention_rate = self.calculate_retention_rate(day, a, b, period)
                                # 使用收入占比计算该付费方案的基础收入
                                period_base_revenue = base_revenue * revenue_prop
                                total_revenue += period_base_revenue * retention_rate
                                
                        revenues.append(total_revenue)
                else:
                    revenues.append(None)
                    
            data[col_name] = revenues
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = '激活日期'
        return df

def calculate_yearly_rpd(df, start_year, end_year):
    """计算每年的RPD和增长率（与原代码相同）"""
    results = {}
    
    for year in range(start_year, end_year + 1):
        mask_cumulative = df.index.year <= year
        users_cumulative = df[mask_cumulative]['激活人数'].sum()
        
        revenue_cumulative = 0
        for idx, row in df[mask_cumulative].iterrows():
            days_until_year_end = (pd.Timestamp(f'{year}-12-31') - idx).days
            revenue_cols = [f'day{i}收入' for i in range(0, days_until_year_end + 1)]
            revenue_cumulative += row[revenue_cols].sum()
        
        rpd_year = revenue_cumulative / users_cumulative
        results[year] = {'RPD': rpd_year}
        
        if year > start_year:
            growth_rate = (rpd_year / results[year-1]['RPD'] - 1) * 100
            results[year]['Growth_Rate'] = growth_rate
    
    return results

def main():
    calculator = RevenueCalculator()
    
    # 示例输入数据格式
    yearly_params = {
        2023: {
            'week': ([0.3, 0.1, 0.05], 0.4),  # (留存率列表, 占比)
            'month': ([0.4, 0.2, 0.1], 0.4),
            'year': ([0.5, 0.3, 0.2], 0.2)
        },
        2024: {
            'week': ([0.35, 0.12, 0.06], 0.4),
            'month': ([0.45, 0.25, 0.12], 0.4),
            'year': ([0.55, 0.35, 0.25], 0.2)
        }
    }
    
    # 创建收入DataFrame
    df = calculator.create_revenue_df(
        active_users=1000,
        start_date='2023-01-01',
        end_date='2024-12-31',
        yearly_params=yearly_params
    )
    
    # 计算RPD
    results = calculate_yearly_rpd(df, 2023, 2024)
    return results, df

if __name__ == "__main__":
    print("请通过streamlit界面运行此程序")
