# 2024-1-9 借助于文心来写的，其实没那么复杂，业务为王，我很认同
# 1-9先暂时如此吧，然后等seer1模型确定之后，保存模型，然后在这里部署
# 1-23 用实践书的企鹅数据做部署，总体就是“建模和输入输出的部署”
# 2024-2-27 这次部署，还是通过重新new一次才可以，回头研究一下
import streamlit as st
import pandas as pd
#from joblib import load #python3.11不支持
# from lifelines import CoxPHFitter
import pickle


# 添加侧边栏图片
logo_path = 'gist.png'
st.sidebar.image(logo_path, width=236)

# 创建侧边栏选项
options = ['Nomogram Models for Prognostic Prediction in Gastric Gastrointestinal Stromal Tumors New Insights from a Global RealWorld Cohort Study','Coming soon']
# selected_option = st.sidebar.selectbox('Select a chart type', options) # 下拉选
selected_option = st.sidebar.radio('Select the Research Paper:', options) #直接选，这个更好


# 根据选择展示不同的图表
if selected_option == 'Nomogram Models for Prognostic Prediction in Gastric Gastrointestinal Stromal Tumors New Insights from a Global RealWorld Cohort Study':
    
    st.header("Please enter your information: ")
 
    # 类别型
    Sex = st.selectbox("Sex", options=["Female", "Male"])
    Race = st.selectbox("Race", options=["Black", "White","Others"])
    Marital_status_at_diagnosis = st.selectbox("Marital status at diagnosis", options=["Single", "Married"])
    Tumor_grade = st.selectbox("Tumor grade", options=["Well/moderately differentiated", "Poorly differentiated/undifferentiated"])
    Tumor_size = st.selectbox("Tumor size", options=["≤2 cm", "2-5cm","5-10cm",">10cm"])
    AJCC_Stage = st.selectbox("AJCC Stage", options=["Ⅰ","Ⅱ","Ⅲ","Ⅳ"])
    Surgery = st.selectbox("Surgery", options=["No Surgery", "Local excision","Radical excision"])
    Regional_nodes_examined = st.selectbox("Regional nodes examined", options=["0", "1-4",">4"])
    # 数值型
    Age_at_diagnosis = st.number_input("Age at diagnosis", min_value=20)
    
    # 模型输入判决赋值
    #年龄
    Age_at_diagnosis = Age_at_diagnosis
    #性别
    if Sex == 'Male':
        Sex_Male = True
    else:
        Sex_Male = False
    # 种族
    if Race == 'Black':
        Race_Black = True
    else:
        Race_Black = False
    #
    if Marital_status_at_diagnosis == "Single":
        Marital_status_at_diagnosis_Single = True
    else:
        Marital_status_at_diagnosis_Single = False
    #
    if Tumor_grade == "Poorly differentiated/undifferentiated":
        Tumor_grade_Poorly_differentiated_undifferentiated = True
    else:
        Tumor_grade_Poorly_differentiated_undifferentiated = False
    #
    if Tumor_size == "5-10cm":
        Tumor_size_5_10cm = True
        Tumor_size_bigger_10cm = False
    elif Tumor_size == ">10cm":
        Tumor_size_bigger_10cm = True
        Tumor_size_5_10cm = False
    else:
        Tumor_size_5_10cm = False
        Tumor_size_bigger_10cm = False
    #
    if AJCC_Stage == "Ⅲ":
        AJCC_Stage_3 = True
        AJCC_Stage_4 = False
    elif AJCC_Stage == "Ⅳ":
        AJCC_Stage_4 = True
        AJCC_Stage_3 = False
    else:
        AJCC_Stage_3 = False
        AJCC_Stage_4 = False
    #
    if Surgery == "No Surgery":
        Surgery_NoSurgery = True
    else:
        Surgery_NoSurgery = False
    # 这个只有css有
    if Regional_nodes_examined == ">4":
        Regional_nodes_examined_bigger_4 = True
    else:
        Regional_nodes_examined_bigger_4 = False
    # 整理输入
    input_os = {'Sex_Male':Sex_Male,'Race_Black':Race_Black,'Marital_status_at_diagnosis_Single':Marital_status_at_diagnosis_Single,'Tumor_grade_Poorly_differentiated_undifferentiated':Tumor_grade_Poorly_differentiated_undifferentiated,'Tumor_size_5_10cm':Tumor_size_5_10cm,'Tumor_size_bigger_10cm':Tumor_size_bigger_10cm,'AJCC_Stage_3':AJCC_Stage_3,'AJCC_Stage_4':AJCC_Stage_4,'Surgery_NoSurgery':Surgery_NoSurgery,'Age_at_diagnosis':Age_at_diagnosis}
    input_css = {'Sex_Male':Sex_Male,'Race_Black':Race_Black,'Marital_status_at_diagnosis_Single':Marital_status_at_diagnosis_Single,'Tumor_grade_Poorly_differentiated_undifferentiated':Tumor_grade_Poorly_differentiated_undifferentiated,'Tumor_size_5_10cm':Tumor_size_5_10cm,'Tumor_size_bigger_10cm':Tumor_size_bigger_10cm,'AJCC_Stage_3':AJCC_Stage_3,'AJCC_Stage_4':AJCC_Stage_4,'Surgery_NoSurgery':Surgery_NoSurgery,'Regional_nodes_examined_bigger_4':Regional_nodes_examined_bigger_4,'Age_at_diagnosis':Age_at_diagnosis}
    #os_df = pd.DataFrame(input_os)
    #css_df = pd.DataFrame(input_css)
    os_df = pd.Series(input_os)
    css_df = pd.Series(input_css)

    # 导入模型
    with open('cph_os.pkl', 'rb') as file:
        cph_os = pickle.load(file)
        os_half = cph_os.predict_survival_function(os_df).loc[6]
        os_1 = cph_os.predict_survival_function(os_df).loc[12]
        os_3 = cph_os.predict_survival_function(os_df).loc[36]
        os_5 = cph_os.predict_survival_function(os_df).loc[60]
        os_7 = cph_os.predict_survival_function(os_df).loc[84]
        os_10 = cph_os.predict_survival_function(os_df).loc[120]
    
    with open('cph_css.pkl', 'rb') as file:
        cph_css = pickle.load(file)
        css_half = cph_css.predict_survival_function(css_df).loc[6]
        css_1 = cph_css.predict_survival_function(css_df).loc[12]
        css_3 = cph_css.predict_survival_function(css_df).loc[36]
        css_5 = cph_css.predict_survival_function(css_df).loc[60]
        css_7 = cph_css.predict_survival_function(css_df).loc[84]
        css_10 = cph_css.predict_survival_function(css_df).loc[120]

    #输出结果
    st.subheader("OS cox nomograms at six-time points: ")
    st.write(f"Half a Year ： {os_half.values}")
    st.write(f"One Year ： {os_1.values}")
    st.write(f"Three Years ： {os_3.values}")
    st.write(f"Five Years ： {os_5.values}")
    st.write(f"Seven Years ： {os_7.values}")
    st.write(f"Ten Years ： {os_10.values}")

    st.subheader("CSS cox nomograms at six-time points: ")
    st.write(f"Half a Year ： {css_half.values}")
    st.write(f"One Year ： {css_1.values}")
    st.write(f"Three Years ： {css_3.values}")
    st.write(f"Five Years ： {css_5.values}")
    st.write(f"Seven Years ： {css_7.values}")
    st.write(f"Ten Years ： {css_10.values}")
elif selected_option == 'Coming soon':

    st.write(f"comming soon...")

# elif selected_option == 'Pie Chart':

#     st.write(f"comming soon")

# else:
#     st.write('Invalid selection')
