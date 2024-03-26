# 2024-1-9 借助于文心来写的，其实没那么复杂，业务为王，我很认同
# 1-9先暂时如此吧，然后等seer1模型确定之后，保存模型，然后在这里部署
# 1-23 用实践书的企鹅数据做部署，总体就是“建模和输入输出的部署”
# 2024-2-27 这次部署，还是通过重新new一次才可以，回头研究一下
# 2024-3-27 部署了seer2的对比，绘图
import streamlit as st
import pandas as pd
import numpy as np
#from joblib import load #python3.11不支持
# from lifelines import CoxPHFitter
import pickle
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Arial' #总是找不到字体，算了

# 添加侧边栏图片
logo_path = 'gist.png'
st.sidebar.image(logo_path, width=236)

# 创建侧边栏选项
options = ['Nomogram Models for Prognostic Prediction in Gastric Gastrointestinal Stromal Tumors: New Insights from a Global RealWorld Cohort Study','Comparative Prognostic Accuracy of Proportional versus Non-Proportional Hazards Models in Gastric Gastrointestinal Stromal Tumors: From Traditional Statistics to Deep Learning','radiomics Comming soon']
# selected_option = st.sidebar.selectbox('Select a chart type', options) # 下拉选
selected_option = st.sidebar.radio('Select the Research Paper:', options) #直接选，这个更好


# 根据选择展示不同的图表
if selected_option == 'Nomogram Models for Prognostic Prediction in Gastric Gastrointestinal Stromal Tumors: New Insights from a Global RealWorld Cohort Study':
    
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
    st.write(f"Half a Year ： {str(os_half.values)}")
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
    st.markdown('''
    **Disclaimer:**
    This research is currently in the laboratory phase. The findings and outcomes presented are preliminary and have not been subjected to the rigorous testing and validation required for clinical application. The use of the information, techniques, or products described herein is at the user's own risk. It is imperative that any potential clinical application be preceded by thorough scientific evaluation and regulatory approval. The authors and affiliated institutions assume no liability for any adverse consequences resulting from the use of the information provided.
    ''')
elif selected_option=='Comparative Prognostic Accuracy of Proportional versus Non-Proportional Hazards Models in Gastric Gastrointestinal Stromal Tumors: From Traditional Statistics to Deep Learning':
    
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
    st.write(f"comming soon...")
    # # 这个版本的pycox似乎有问题，笔记本可以，云端不行
    # tt = np.array([[0.57872546, 0.        , 0.        , 1.        , 0.        ,
    #    1.        , 0.        , 1.        , 1.        , 0.        ]], dtype=np.float32)
    # with open('deepcox.pkl', 'rb') as file:
    #     deepcox = pickle.load(file)
    #     _ = deepcox.compute_baseline_hazards() #计算基准风险函数
    #     surv = deepcox.predict_surv_df(tt) #计算生存函数
    #     ax=surv.plot()
    #     plt.ylabel('S(t | x)')
    #     _ = plt.xlabel('Time')
    #     ax.get_legend().remove
    # 下面暂时用coxph做一个展示，后续排查之后再把其余加上去，ps：另外rsf由于pickle文件过大无法上传，后续部署个人服务器会加上
    # 你提供的字典
    patient_data = {
        'Sex_Male': Sex_Male,
        'Race_Black': Race_Black,
        'Marital_status_at_diagnosis_Single': Marital_status_at_diagnosis_Single,
        'Tumor_size_5_10cm': Tumor_size_5_10cm,
        'Tumor_size_bigger_10cm':Tumor_size_bigger_10cm,
        'AJCC_Stage_III': AJCC_Stage_3,
        'AJCC_Stage_IV': AJCC_Stage_4,
        'Surgery_NoSurgery': Surgery_NoSurgery,
        'Regional_nodes_examined_bigger_4': Regional_nodes_examined_bigger_4,
        'Age_at_diagnosis': Age_at_diagnosis,
    }
    # 将字典转换为pandas Series然后为dataframe
    patient_series = (pd.Series(patient_data)).to_frame()
    patient_df = patient_series.T
    # 开始预测
    st.subheader("Probability of Survival on Proportional Hazards ")
    with open('cph.pkl', 'rb') as file:
        cph = pickle.load(file)
        survival_function_df = cph.predict_survival_function(patient_df)
        survival_function_df = survival_function_df.rename(columns={0:'Proportional Hazards'})
        fig, ax = plt.subplots(figsize=(4, 3))
        survival_function_df.plot(ax=ax, color='#FF8C00')

        ax.set_ylabel(r"Probability of Survival $\hat{S}(t)$")
        ax.set_xlabel("time $t$")
        ax.legend(loc="best")

        st.pyplot(fig)

    
        # cph.predict_survival_function(patient_df).rename(columns={0:'Proportional Hazards'}).plot(color='#FF8C00')
        # plt.ylabel(r"Probability of Survival $\hat{S}(t)$")
        # plt.xlabel("time $t$")
        # plt.legend(loc="best")
        # plt.show()
    st.subheader("Probability of Survival on Non-Proportional Hazards ")
    with open('coxboost.pkl', 'rb') as file:
        coxboost = pickle.load(file)
        pred_surv = coxboost.predict_survival_function(patient_df)
        fig, ax = plt.subplots(figsize=(4, 3))
        time_points = np.arange(1, 251)
        for i, surv_func in enumerate(pred_surv):
            ax.step(time_points, surv_func(time_points), where="post", label="Non-Proportional Hazards",color='#0072BD')
        ax.set_ylabel(r"Probability of Survival $\hat{S}(t)$")
        ax.set_xlabel("time $t$")
        ax.legend(loc="best")
        st.pyplot(fig)
    st.markdown('''
    **Disclaimer:**
    This research is currently in the laboratory phase. The findings and outcomes presented are preliminary and have not been subjected to the rigorous testing and validation required for clinical application. The use of the information, techniques, or products described herein is at the user's own risk. It is imperative that any potential clinical application be preceded by thorough scientific evaluation and regulatory approval. The authors and affiliated institutions assume no liability for any adverse consequences resulting from the use of the information provided.
    ''')
elif selected_option == 'radiomics Comming soon':

    st.write(f"comming soon...")

# elif selected_option == 'Pie Chart':

#     st.write(f"comming soon")

# else:
#     st.write('Invalid selection')
