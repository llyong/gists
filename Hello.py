# 2024-1-9 借助于文心来写的，其实没那么复杂，业务为王，我很认同
# 1-9先暂时如此吧，然后等seer1模型确定之后，保存模型，然后在这里部署
import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt

# 加载数据
# df = pd.read_csv('data.csv')
# 添加logo
# 设置Logo路径，请替换为你的Logo图片路径
logo_path = 'python.jpg'

# 在侧边栏中展示Logo
st.sidebar.image(logo_path, width=200)

# 创建侧边栏选项
options = ['Line Plot', 'Bar Chart', 'Pie Chart']
# selected_option = st.sidebar.selectbox('Select a chart type', options) # 下拉选
selected_option = st.sidebar.radio('Select a chart type', options) #直接选，这个更好

# 根据选择展示不同的图表
if selected_option == 'Line Plot':
    st.write('This is a line plot:')

elif selected_option == 'Bar Chart':
    st.write('This is a bar chart:')

elif selected_option == 'Pie Chart':
    st.write('This is a pie chart:')

else:
    st.write('Invalid selection')
