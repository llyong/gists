# 2024-1-9 借助于文心来写的，其实没那么复杂，业务为王，我很认同
# 1-9先暂时如此吧，然后等seer1模型确定之后，保存模型，然后在这里部署
# 1-23 用实践书的企鹅数据做部署，总体就是“建模和输入输出的部署”

import streamlit as st
import pandas as pd
from joblib import load


# 添加侧边栏图片
logo_path = 'gist.png'
st.sidebar.image(logo_path, width=236)

# 创建侧边栏选项
options = ['Nomogram Models for Prognostic Prediction in Gastric Gastrointestinal Stromal Tumors New Insights from a Global RealWorld Cohort Study','seer1 GISTs', 'seer2 GISTs', 'Pie Chart']
# selected_option = st.sidebar.selectbox('Select a chart type', options) # 下拉选
selected_option = st.sidebar.radio('Select the Research Paper:', options) #直接选，这个更好


# 根据选择展示不同的图表
if selected_option == 'Nomogram Models for Prognostic Prediction in Gastric Gastrointestinal Stromal Tumors New Insights from a Global RealWorld Cohort Study':
    # 导入模型
    cph_os = load('cph_os.joblib')
    cph_css = load('cph_css.joblib')
    #['Sex_Male','Race_Black','Marital_status_at_diagnosis_Single',
           # 'Tumor_grade_Poorly_differentiated_undifferentiated','Tumor_size_5_10cm','Tumor_size_bigger_10cm',
            #'AJCC_Stage_3','AJCC_Stage_4','Surgery_NoSurgery','Regional_nodes_examined_bigger_4','Age_at_diagnosis']
 
    # 类别型
    sex = st.selectbox("Sex", options=["Female", "Male"])
    Race = st.selectbox("Race", options=["Black", "White","Others"])
    Marital_status_at_diagnosis = st.selectbox("Marital status at diagnosis", options=["Single", "Married"])

    # 数值型
    bill_length = st.number_input("Bill Length (mm)", min_value=0)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
    body_mass = st.number_input("Body Mass (g)", min_value=0)
    island_biscoe, island_dream, island_torgerson = 0, 0, 0

    # 模型输入
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
