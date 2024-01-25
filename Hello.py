# 2024-1-9 借助于文心来写的，其实没那么复杂，业务为王，我很认同
# 1-9先暂时如此吧，然后等seer1模型确定之后，保存模型，然后在这里部署
# 1-23 用实践书的企鹅数据做部署，总体就是“建模和输入输出的部署”

import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
# import pickle

# 添加侧边栏图片
logo_path = 'gist.png'
st.sidebar.image(logo_path, width=233)

# 创建侧边栏选项
options = ['seer1 GISTs', 'seer2 GISTs', 'Pie Chart']
# selected_option = st.sidebar.selectbox('Select a chart type', options) # 下拉选
selected_option = st.sidebar.radio('Select the Research Paper:', options) #直接选，这个更好


# 根据选择展示不同的图表
if selected_option == 'seer1 GISTs':
    # rf_pickle = open('random_forest_penguin.pickle', 'rb')
    # map_pickle = open('output_penguin.pickle', 'rb')
    # rfc = pickle.load(rf_pickle)
    # unique_penguin_mapping = pickle.load(map_pickle)
    # rf_pickle.close()
    # map_pickle.close()

    island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgerson"])
    sex = st.selectbox("Sex", options=["Female", "Male"])
    bill_length = st.number_input("Bill Length (mm)", min_value=0)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
    body_mass = st.number_input("Body Mass (g)", min_value=0)
    island_biscoe, island_dream, island_torgerson = 0, 0, 0
    if island == 'Biscoe':
        island_biscoe = 1
    elif island == 'Dream':
        island_dream = 1
    elif island == 'Torgerson':
        island_torgerson = 1
    sex_female, sex_male = 0, 0
    if sex == 'Female':
        sex_female = 1
    elif sex == 'Male':
        sex_male = 1
    # new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass, island_biscoe, island_dream,
    #                                island_torgerson, sex_female, sex_male]])
    # prediction_species = unique_penguin_mapping[new_prediction][0]
    # st.write(f"We predict your penguin is of the {prediction_species} species")
    Result = bill_length+ bill_depth+ flipper_length+ body_mass+ island_biscoe+ island_dream+island_torgerson+ sex_female+ sex_male
    st.write(f"Result ： {Result}")


elif selected_option == 'seer2 GISTs':

    st.write(f"comming soon")

elif selected_option == 'Pie Chart':

    st.write(f"comming soon")

else:
    st.write('Invalid selection')
