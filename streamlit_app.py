import streamlit as st
import pandas as pd
import numpy as np
from advanced_rpd import RevenueCalculator, calculate_yearly_rpd, log_power_function, fit_revenue_parameters
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import lru_cache

# 设置页面配置（必须在最前面）
st.set_page_config(
    page_title="RPD计算工具",
    page_icon="📊",
    layout="wide"
)

def create_input_section(year, year_number):
    """创建输入部分
    year: 实际年份
    year_number: 产品上线后的第几年
    """
    st.subheader(f"{year}年（产品上线第{year_number}年）参数设置")
    
    params = {}
    
    # 收入占比设置
    st.write("各付费方案收入占比设置：")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        week_prop = st.number_input(f"周付费收入占比", 0.0, 1.0, 0.2, key=f"week_prop_{year}")
    with col2:
        month_prop = st.number_input(f"月付费收入占比", 0.0, 1.0, 0.2, key=f"month_prop_{year}")
    with col3:
        quarter_prop = st.number_input(f"季付费收入占比", 0.0, 1.0, 0.2, key=f"quarter_prop_{year}")
    with col4:
        year_prop = st.number_input(f"年付费收入占比", 0.0, 1.0, 0.4, key=f"year_prop_{year}")
    
    # 检查占比之和是否为1
    total_prop = week_prop + month_prop + quarter_prop + year_prop
    if abs(total_prop - 1.0) > 0.001:
        st.warning(f"各付费方案收入占比之和应为1，当前为{total_prop:.2f}")
    
    st.markdown("---")  # 添加分隔线
    
    # 周付费续订率参数
    st.write("周付费续订率：")
    col1, col2, col3 = st.columns(3)
    with col1:
        week_1 = st.number_input(f"第1周续订率", 0.0, 1.0, 0.89, key=f"week_1_{year}")
    with col2:
        week_3 = st.number_input(f"第3周续订率", 0.0, 1.0, 0.76, key=f"week_3_{year}")
    with col3:
        week_7 = st.number_input(f"第7周续订率", 0.0, 1.0, 0.60, key=f"week_7_{year}")
    
    # 月付费续订率参数
    st.write("月付费续订率：")
    col1, col2, col3 = st.columns(3)
    with col1:
        month_1 = st.number_input(f"第1月续订率", 0.0, 1.0, 0.85, key=f"month_1_{year}")
    with col2:
        month_3 = st.number_input(f"第3月续订率", 0.0, 1.0, 0.70, key=f"month_3_{year}")
    with col3:
        month_7 = st.number_input(f"第7月续订率", 0.0, 1.0, 0.55, key=f"month_7_{year}")
    
    # 季付费续订率参数
    st.write("季付费续订率：")
    col1, col2, col3 = st.columns(3)
    with col1:
        quarter_1 = st.number_input(f"第1季续订率", 0.0, 1.0, 0.82, key=f"quarter_1_{year}")
    with col2:
        quarter_3 = st.number_input(f"第3季续订率", 0.0, 1.0, 0.65, key=f"quarter_3_{year}")
    with col3:
        quarter_4 = st.number_input(f"第4季续订率", 0.0, 1.0, 0.60, key=f"quarter_4_{year}")
    
    # 年付费续订率参数
    st.write("年付费续订率：")
    col1, col2, col3 = st.columns(3)
    with col1:
        year_1 = st.number_input(f"第1年续订率", 0.0, 1.0, 0.80, key=f"year_1_{year}")
    with col2:
        year_2 = st.number_input(f"第2年续订率", 0.0, 1.0, 0.70, key=f"year_2_{year}")
    with col3:
        year_3 = st.number_input(f"第3年续订率", 0.0, 1.0, 0.60, key=f"year_3_{year}")
    
    st.markdown("---")  # 添加分隔线
    
    params = {
        'week': ([week_1, week_3, week_7], week_prop),
        'month': ([month_1, month_3, month_7], month_prop),
        'quarter': ([quarter_1, quarter_3, quarter_4], quarter_prop),
        'year': ([year_1, year_2, year_3], year_prop)
    }
    
    return params

def plot_retention_curves(yearly_params, launch_date, is_yearly_params):
    """绘制续订率曲线
    yearly_params: 年度参数字典
    launch_date: 产品上线日期
    is_yearly_params: 参数设置模式
    """
    # 计算从上线到今天的时间跨度
    # days_since_launch = (datetime.now() - launch_date).days
    
    # 计算最大观察期（以周期为单位）
    max_weeks = 104  # 2年
    max_months = 24  # 2年
    max_quarters = 20  # 5年
    max_years = 10 # 10年
    
    # 创建四个子图
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=("周付费用户续订率曲线", "月付费用户续订率曲线", "季付费用户续订率曲线", "年付费用户续订率曲线"),
        vertical_spacing=0.10
    )

    # 定义观测点
    points = {
        'week': np.array([1, 3, 7]),
        'month': np.array([1, 3, 7]),
        'quarter': np.array([1, 3, 4]),
        'year': np.array([1, 2, 3])
    }
    
    # 定义最大观察期
    max_periods = {
        'week': max_weeks,
        'month': max_months,
        'quarter': max_quarters,
        'year': max_years
    }
    
    # 定义颜色列表
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # 为每种付费方式绘制曲线
    for period_idx, period in enumerate(['week', 'month', 'quarter', 'year'], 1):
        # 生成连续的周期点用于绘制平滑曲线
        x_smooth = np.linspace(1, max_periods[period], 100)
        
        if is_yearly_params == "所有年份相同":
            # 获取观测点数据
            rates = yearly_params[launch_date.year][period][0]
            
            # 拟合曲线
            a, b = fit_revenue_parameters(points[period], np.array(rates))
            y_smooth = np.exp(log_power_function(x_smooth, a, b))
            
            # 绘制观测点
            fig.add_trace(
                go.Scatter(
                    x=points[period],
                    y=rates,
                    mode='markers',
                    name=f"{period}付费-观测点",
                    marker=dict(size=8, color='blue'),
                    showlegend=True if period_idx == 1 else False
                ),
                row=period_idx, col=1
            )
            
            # 绘制拟合曲线
            fig.add_trace(
                go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    name=f"{period}付费-拟合曲线",
                    line=dict(color='blue'),
                    showlegend=True if period_idx == 1 else False
                ),
                row=period_idx, col=1
            )
        else:
            # 为每年绘制不同的曲线
            for i, year in enumerate(yearly_params.keys()):
                rates = yearly_params[year][period][0]
                a, b = fit_revenue_parameters(points[period], np.array(rates))
                y_smooth = np.exp(log_power_function(x_smooth, a, b))
                
                # 绘制观测点
                fig.add_trace(
                    go.Scatter(
                        x=points[period],
                        y=rates,
                        mode='markers',
                        name=f"{year}年-观测点",
                        marker=dict(size=8, color=colors[i]),
                        showlegend=True if period_idx == 1 else False
                    ),
                    row=period_idx, col=1
                )
                
                # 绘制拟合曲线
                fig.add_trace(
                    go.Scatter(
                        x=x_smooth,
                        y=y_smooth,
                        mode='lines',
                        name=f"{year}年-拟合曲线",
                        line=dict(color=colors[i]),
                        showlegend=True if period_idx == 1 else False
                    ),
                    row=period_idx, col=1
                )
    
    # 更新布局
    fig.update_layout(height=1000)
    
    # 更新x轴标签
    fig.update_xaxes(title_text="周数", row=1, col=1)
    fig.update_xaxes(title_text="月数", row=2, col=1)
    fig.update_xaxes(title_text="季数", row=3, col=1)
    fig.update_xaxes(title_text="年数", row=4, col=1)
    
    # 更新y轴标签
    for i in range(1, 5):
        fig.update_yaxes(title_text="续订率", row=i, col=1)
    
    return fig

# 添加缓存装饰器
@st.cache_data
def create_revenue_df_cached(active_users, start_date, end_date, yearly_params):
    calculator = RevenueCalculator()
    return calculator.create_revenue_df(active_users, start_date, end_date, yearly_params)

@st.cache_data
def calculate_yearly_rpd_cached(df, start_year, end_year):
    return calculate_yearly_rpd(df, start_year, end_year)

@st.cache_data
def plot_retention_curves_cached(yearly_params, launch_date, is_yearly_params):
    return plot_retention_curves(yearly_params, launch_date, is_yearly_params)

def main():
    st.title("基于续订率的RPD计算工具")
    
    # 添加侧边栏
    with st.sidebar:
        st.header("基础参数设置")
        launch_date = pd.Timestamp(st.date_input(
            "产品上线日期",
            value=datetime(2024, 1, 1),
            min_value=datetime(2000, 1, 1),
            max_value=datetime.now()
        ))
        forecast_years = st.number_input("预测年数", 1, 100, 10)
        is_yearly_params = st.radio(
            "参数设置模式",
            ["每年单独设置", "所有年份相同"],
            index=1
        )
    
    # 计算年份范围
    start_year = launch_date.year
    end_year = start_year + forecast_years - 1
    
    # 为每年创建参数输入区
    yearly_params = {}
    
    if is_yearly_params == "所有年份相同":
        st.header("续订率参数设置（适用所有年份）")
        base_params = create_input_section(start_year, 1)
        for year in range(start_year, end_year + 1):
            yearly_params[year] = base_params
    else:
        for i, year in enumerate(range(start_year, end_year + 1), 1):
            yearly_params[year] = create_input_section(year, i)
    
    # 计算按钮
    if st.button("计算RPD"):
        with st.spinner("正在计算，请稍候..."):
            # 使用缓存的计算函数
            df = create_revenue_df_cached(
                active_users=1000,
                start_date=launch_date.strftime("%Y-%m-%d"),
                end_date=f"{end_year}-12-31",
                yearly_params=yearly_params
            )
            
            # 计算RPD
            results = calculate_yearly_rpd_cached(df, start_year, end_year)
            
            # 显示结果
            st.header("计算结果")
            
            # 创建结果DataFrame
            rpd_data = []
            for i, (year, data) in enumerate(results.items(), 1):
                row = {
                    '年份': f'{year}年（产品上线第{i}年）',
                    'RPD': f"{data['RPD']:.2f}",
                    '增长率': f"{data.get('Growth_Rate', '-'):.2f}%" if 'Growth_Rate' in data else '-'
                }
                rpd_data.append(row)
            
            result_df = pd.DataFrame(rpd_data)
            st.table(result_df)
            
            # 绘制RPD趋势图
            rpd_values = [data['RPD'] for data in results.values()]
            chart_df = pd.DataFrame({
                '年份': [f'第{i}年' for i in range(1, len(rpd_values) + 1)],
                'RPD': rpd_values
            })
            st.line_chart(chart_df.set_index('年份'))
            
            # 添加基本信息展示
            st.subheader("基本信息")
            st.write(f"产品上线日期：{launch_date.strftime('%Y-%m-%d')}")
            st.write(f"参数设置模式：{is_yearly_params}")
            
            # 提供下载详细数据的功能
            st.download_button(
                label="下载详细数据",
                data=df.to_csv(index=True).encode('utf-8'),
                file_name='rpd_detailed_data.csv',
                mime='text/csv'
            )
            
            # 添加续订率曲线
            st.subheader("续订率曲线")
            fig = plot_retention_curves_cached(yearly_params, launch_date, is_yearly_params)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()