# 2024-1-9 开始的，后来搁置一段时间
# 2024-2-27 部署第一篇，还是通过重新new一次才可以，回头研究一下
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
options = ['Evaluating Nomogram Models for Predicting Survival Outcomes in Gastric Gastrointestinal Stromal Tumors with SEER Database Analysis','Comparative Prognostic Accuracy of Proportional versus Non-Proportional Hazards Models in Gastric Gastrointestinal Stromal Tumors: From Traditional Statistics to Deep Learning','radiomics Comming soon']
# selected_option = st.sidebar.selectbox('Select a chart type', options) # 下拉选
selected_option = st.sidebar.radio('Select the Research Paper:', options) #直接选，这个更好

##########################################################    1    #############################################################
# 根据选择展示不同的图表
if selected_option == 'Evaluating Nomogram Models for Predicting Survival Outcomes in Gastric Gastrointestinal Stromal Tumors with SEER Database Analysis':
    image_file = "seer1.jpg"
    st.title(":tada: Evaluating nomogram models for predicting survival outcomes in gastric gastrointestinal stromal tumors with SEER database analysis")
    url = "https://pubmed.ncbi.nlm.nih.gov/38769376/"
    text_to_display = "DOI: 10.1038/s41598-024-62353-z"
    st.markdown(f"[{text_to_display}]({url})", unsafe_allow_html=True)
    
    st.image(image_file, caption='GraphicalAbstract', use_column_width=True)
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

##########################################################    2    #############################################################

elif selected_option=='Comparative Prognostic Accuracy of Proportional versus Non-Proportional Hazards Models in Gastric Gastrointestinal Stromal Tumors: From Traditional Statistics to Deep Learning':
    # 这里分三页

    st.title(":apple: Comparative Prognostic Accuracy of Proportional versus Non-Proportional Hazards Models in Gastric Gastrointestinal Stromal Tumors: From Traditional Statistics to Deep Learning")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Article Overview","Information input","Proportional Hazards Model","Non-Proportional Hazards Model","Python Code"])

    with tab1:
        st.image('seer2.png', width=876)
    with tab2:
        st.header("Please enter the information, and the real-time results will be displayed on the 'Proportional Hazards Model' and 'Non-Proportional Hazards Model' page : ")
 
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
        
    with tab3:
        # 开始预测
        st.subheader("Probability of Survival on Proportional Hazards ")
        with open('cph.pkl', 'rb') as file:
            cph = pickle.load(file)
            survival_function_df = cph.predict_survival_function(patient_df)
            survival_function_df = survival_function_df.rename(columns={0:'Proportional Hazards'})
            fig, ax = plt.subplots(figsize=(4, 3))
            survival_function_df.plot(ax=ax, color='#FF8C00')
    
            ax.set_ylabel(r"Probability of Survival $\hat{S}(t)$")
            ax.set_xlabel("time $t$ (months)")
            ax.legend(loc="best")
    
            st.pyplot(fig)
    
        
            # cph.predict_survival_function(patient_df).rename(columns={0:'Proportional Hazards'}).plot(color='#FF8C00')
            # plt.ylabel(r"Probability of Survival $\hat{S}(t)$")
            # plt.xlabel("time $t$")
            # plt.legend(loc="best")
            # plt.show()
        # st.subheader("Probability of Survival on Non-Proportional Hazards ")
        # with open('coxboost.pkl', 'rb') as file:
        #     coxboost = pickle.load(file)
        #     pred_surv = coxboost.predict_survival_function(patient_df)
        #     fig, ax = plt.subplots(figsize=(4, 3))
        #     time_points = np.arange(1, 251)
        #     for i, surv_func in enumerate(pred_surv):
        #         ax.step(time_points, surv_func(time_points), where="post", label="Non-Proportional Hazards",color='#0072BD')
        #     ax.set_ylabel(r"Probability of Survival $\hat{S}(t)$")
        #     ax.set_xlabel("time $t$ (months)")
        #     ax.legend(loc="best")
        #     st.pyplot(fig)
        st.markdown('''
        **Disclaimer:**
        This research is currently in the laboratory phase. The findings and outcomes presented are preliminary and have not been subjected to the rigorous testing and validation required for clinical application. The use of the information, techniques, or products described herein is at the user's own risk. It is imperative that any potential clinical application be preceded by thorough scientific evaluation and regulatory approval. The authors and affiliated institutions assume no liability for any adverse consequences resulting from the use of the information provided.
        ''')
    with tab4:

        # 开始预测

        st.subheader("Probability of Survival on Non-Proportional Hazards ")
        with open('coxboost.pkl', 'rb') as file:
            coxboost = pickle.load(file)
            pred_surv = coxboost.predict_survival_function(patient_df)
            fig, ax = plt.subplots(figsize=(4, 3))
            time_points = np.arange(1, 251)
            for i, surv_func in enumerate(pred_surv):
                ax.step(time_points, surv_func(time_points), where="post", label="Non-Proportional Hazards",color='#0072BD')
            ax.set_ylabel(r"Probability of Survival $\hat{S}(t)$")
            ax.set_xlabel("time $t$ (months)")
            ax.legend(loc="best")
            st.pyplot(fig)
        st.markdown('''
        **Disclaimer:**
        This research is currently in the laboratory phase. The findings and outcomes presented are preliminary and have not been subjected to the rigorous testing and validation required for clinical application. The use of the information, techniques, or products described herein is at the user's own risk. It is imperative that any potential clinical application be preceded by thorough scientific evaluation and regulatory approval. The authors and affiliated institutions assume no liability for any adverse consequences resulting from the use of the information provided.
        ''')
    with tab5:
        #代码展示
        st.image('seer23.jpg',caption='Schematic diagram of code and article correspondence')
        st.subheader('Data Preprocessing')
        # st.code('''''',language='python',line_numbers=True)
        st.code('''# Perform KM (Kaplan-Meier) analysis based on the raw data obtained from the SEER database

import pandas as pd
df_os=pd.read_excel('seerdata4paper1os.xlsx')
df_css=pd.read_excel('seerdata4paper1css.xlsx')

cat_features = ['Sex', 'Race',
       'Marital status at diagnosis', 'Tumor location', 'Tumor grade',
       'Tumor size', 'AJCC Stage', 'Mitotic rate', 'Surgery',
       'Regional nodes examined', 'Chemotherapy']
num_features = ['Age at diagnosis (years)']
all_features = ['Age at diagnosis (years)', 'Sex', 'Race',
       'Marital status at diagnosis', 'Tumor location', 'Tumor grade',
       'Tumor size', 'AJCC Stage', 'Mitotic rate', 'Surgery',
       'Regional nodes examined', 'Chemotherapy']
y_labels=['Survival (months)','COD'] # In COD (Cause of Death), death is represented as True, while still alive is represented as False.''',language='python',line_numbers=True)

        st.code('''# KM plot for binary categorical variables

def bin_km(key,value):
    column = key
    features = value
    
    plt.figure() 

    for treatment_type in features:
        mask_treat = df_css[column] == treatment_type
        time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(
            df_css["COD"][mask_treat],
            df_css["Survival (months)"][mask_treat],
            conf_type="log-log",
        )
        
        plt.step(time_treatment, survival_prob_treatment, where="post", label=f"{column}= {treatment_type}")
        plt.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post")

    durations_A = df_css["Survival (months)"][df_css[column] == features[0]]
    durations_B = df_css["Survival (months)"][df_css[column] == features[1]]
    event_observed_A = df_css["COD"][df_css[column] == features[0]]
    event_observed_B = df_css["COD"][df_css[column] == features[1]]

#     result = logrank_test(durations_A,durations_B,event_observed_A,event_observed_B)
#     result_p = result.p_value
#     result_tex = '%.2f'%result_p

    plt.ylim(0, 1)
    plt.ylabel("GIST specific survival")
    plt.xlabel("months")
#     plt.annotate(f'p-value = {result_tex}', xy=(25, 0.3)) # Plot p-values and positions
    plt.legend(loc="best")
    plt.savefig(f'figs/bin_{key}.pdf',bbox_inches='tight',dpi=300)
    
    
# KM plot for multiclass categorical variables
def multi_km(key,value):
    column = key
    features = value
    plt.figure()
    
    for treatment_type in features:
        mask_treat = df_css[column] == treatment_type
        time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(
            df_css["COD"][mask_treat],
            df_css["Survival (months)"][mask_treat],
            conf_type="log-log",
        )

        plt.step(time_treatment, survival_prob_treatment, where="post", label=f"{column}= {treatment_type}")
        plt.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post")


    event_durations = df_css["Survival (months)"]
    groups = df_css[column].map(lambda i: i in features) # Check if the value is within the features list
    event_observed = df_css["COD"]

#     results = multivariate_logrank_test(event_durations,groups,event_observed) #multiclass categorical variables
#     result_p = result.p_value
#     result_tex = '%.2f'%result_p

    plt.ylim(0, 1)
    plt.ylabel("GIST specific survival")
    plt.xlabel("months")
#     plt.annotate(f'p-value = {result_tex}', xy=(25, 0.3)) # Plot p-values and positions
    plt.legend(loc="best")

    plt.savefig(f'figs/multi_{key}.pdf',bbox_inches='tight',dpi=300)''',language='python',line_numbers=True)
        st.code('''# Integrate the column vector feature names and the specific value ranges contained in features into a dictionary format.

bin_dict = {'Sex':[ "Female", "Male"],'Marital status at diagnosis':[ "Married", "Single"],'Tumor grade':[ "Well/moderately differentiated", "Poorly differentiated/undifferentiated"],'Mitotic rate': [">5/5mm2 HP", "≤5/5mm2 HPF"],'Chemotherapy': ["No/Unknown", "Yes"]}
multi_dict = {'Race': ["White", "Black",'others'], 'Tumor location': ['Antrum and Pylorus','Body','Fundus', 'Cardia'], 'Tumor size': ['≤2 cm', '>10cm', '2-5cm', '5-10cm'], 'AJCC Stage': ['Ⅱ' ,'Ⅳ', 'Ⅰ' , 'Ⅲ'], 'Surgery': ['Radical excision','No Surgery', 'Local excision'], 'Regional nodes examined': [0, '>4','1-4']}

import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
# Times New Roman
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

for key,value in bin_dict.items():
    bin_km(key,value)
for key,value in multi_dict.items():
    multi_km(key,value)
''',language='python',line_numbers=True)
        st.code('''# Replace spaces in the column names of the raw data with underscores and remove parentheses.
# Import the data, with missing values represented by NaN
import pandas as pd
import numpy as np

df_os = pd.read_excel('seerdata4paper2os.xlsx') 
df_css = pd.read_excel('seerdata4paper2css.xlsx') 

# Remove unused columns
df_os = df_os.drop(columns=['Unnamed: 0','Survival_months','COD to site recode ICD-O-3 2023 Revision','COD'])
df_css = df_css.drop(columns=['Unnamed: 0','Survival_months','COD to site recode ICD-O-3 2023 Revision','COD'])

# Set categorical features' data as category
nan_cols = ['Sex', 'Race',
       'Marital_status_at_diagnosis', 'Tumor_location', 'Tumor_grade',
       'Tumor_size', 'AJCC_Stage', 'Mitotic_rate', 'Surgery',
       'Regional_nodes_examined', 'Chemotherapy']
for col in nan_cols:
    df_os[col] = df_os[col].astype('category')
    df_css[col] = df_css[col].astype('category')

# Categories are mapped to numbers - append 0 to the file suffix
df_os0 = df_os.copy(deep=True)
df_css0 = df_css.copy(deep=True)
for data in [df_os0,df_css0]:
    data['Sex']=data['Sex'].map({'Female':0,'Male':1})
    data['Race']=data['Race'].map({'others':0,'White':1,'Black':2})
    data['Marital_status_at_diagnosis']=data['Marital_status_at_diagnosis'].map({'Single':0,'Married':1})
    data['Tumor_location']=data['Tumor_location'].map({'Cardia':0,'Fundus':0,'Body':1,'Antrum and Pylorus':2})
    data['Tumor_grade']=data['Tumor_grade'].map({'Well/moderately differentiated':0,'Poorly differentiated/undifferentiated':1})
    data['Tumor_size']=data['Tumor_size'].map({'≤2 cm':0,'2-5cm':1,'5-10cm':2,'>10cm':3})
    data['AJCC_Stage']=data['AJCC_Stage'].map({'Ⅰ':0,'Ⅱ':1,'Ⅲ':2,'Ⅳ':3})
    data['Mitotic_rate']=data['Mitotic_rate'].map({'≤5/5mm2 HPF':0,'>5/5mm2 HP':1})
    data['Surgery']=data['Surgery'].map({'No Surgery':0,'Local excision':1,'Radical excision':2})
    data['Regional_nodes_examined']=data['Regional_nodes_examined'].map({0:0,'1-4':1,'>4':2})
    data['Chemotherapy']=data['Chemotherapy'].map({'No/Unknown':0,'Yes':1})

# The Tumor_location category has been changed and needs to be reset as category
col = 'Tumor_location'
df_os0[col] = df_os0[col].astype('category')
df_css0[col] = df_css0[col].astype('category')
df_css0.dtypes''',language='python',line_numbers=True)
        st.code('''# Use MICE with catboost for missing value imputation, make sure to import misscatboosts first
# https://github.com/llyong/MissCatboosts
from misscatboosts.misscatboosts import MissCatboosts

mc = MissCatboosts()
data_imputed = mc.fit_transform(
    X=df_css0,
    categorical=["sex", "Race", "Marital_status_at_diagnosis","Tumor_location","Tumor_grade",
                 "Tumor_size","AJCC_Stage","Mitotic_rate","Surgery","Regional_nodes_examined","Chemotherapy"]
)
''',language='python',line_numbers=True)
        st.code('''# Convert numerical variables to categorical
column_names = ['Age_at_diagnosis', 'Sex', 'Race', 'Marital_status_at_diagnosis',
       'Tumor_location', 'Tumor_grade', 'Tumor_size', 'AJCC_Stage',
       'Mitotic_rate', 'Surgery', 'Regional_nodes_examined', 'Chemotherapy'] 
df_css0_complete =  pd.DataFrame(data_imputed,columns = column_names)


for data in [df_css0_complete]:
    data['Sex']=data['Sex'].map({0:'Female',1:'Male'})
    data['Race']=data['Race'].map({0:'others',1:'White',2:'Black'})
    data['Marital_status_at_diagnosis']=data['Marital_status_at_diagnosis'].map({0:'Single',1:'Married'})
    data['Tumor_location']=data['Tumor_location'].map({0:'Cardia_Fundus',1:'Body',2:'Antrum_Pylorus'})
    data['Tumor_grade']=data['Tumor_grade'].map({0:'Well_moderately_differentiated',1:'Poorly_differentiated_undifferentiated'})
    data['Tumor_size']=data['Tumor_size'].map({0:'smaller_2cm',1:'2_5cm',2:'5_10cm',3:'bigger_10cm'})
    data['AJCC_Stage']=data['AJCC_Stage'].map({0:1,1:2,2:3,3:4})
    data['Mitotic_rate']=data['Mitotic_rate'].map({0:'smaller_5HPF',1:'bigger_5HPF'})
    data['Surgery']=data['Surgery'].map({0:'NoSurgery',1:'Local_excision',2:'Radical_excision'})
    data['Regional_nodes_examined']=data['Regional_nodes_examined'].map({0:0,1:'1to4',2:'bigger_4'})
    data['Chemotherapy']=data['Chemotherapy'].map({0:'No_Unknown',1:'Yes'})

df_css0_complete['Survival_months'] = df_css['Survival_months']
df_css0_complete['COD'] = df_css['COD']
df_css0_complete.to_csv('df_css0_complete_11.csv',index=False) 
''',language='python',line_numbers=True)
        st.subheader('Univariate And Multivariate Analysis')
        st.code('''# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

plt.rcParams['legend.fontsize'] = 18    # Font size should not be too small
plt.rcParams['font.size'] = 18 # Adjust the number size on the x and y axes

FIGSIZE = (3.5,2.5)
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from matplotlib import pyplot as plt
from lifelines import CoxPHFitter
import numpy as np

df_css0_complete_11=pd.read_csv('df_css0_complete_11.csv') ''',language='python',line_numbers=True)
        st.code('''# Univariate Analysis
# Scaled Schoenfeld residuals plots and proportional hazard test

# Dummy processing
# In the DataPreprocessing program, find the category with the lowest risk level for each variable

# Perform one-hot encoding for all categories first.
data_cat = pd.get_dummies(df_css0_complete_11.drop(columns=['Age_at_diagnosis','Survival_months', 'COD'],axis=1)) 
data_cat

# After processing the categorical variables, they are recombined with the continuous features.
final_data = pd.concat([data_cat,df_css0_complete_11.loc[:,['Age_at_diagnosis','Survival_months', 'COD']]],axis=1)
final_data.columns
''',language='python',line_numbers=True)
        st.code('''# 2. Perform Univariate Analysis after removing the reference variable.

# After one-hot encoding categorical variables, the resulting features can be enclosed in square brackets again.
col_signal_features = [['Sex_Female', 'Sex_Male'], 
                       ['Race_Black', 'Race_White', 'Race_others'],
                       ['Marital_status_at_diagnosis_Married','Marital_status_at_diagnosis_Single'], 
                       ['Tumor_location_Antrum_Pylorus','Tumor_location_Body', 'Tumor_location_Cardia_Fundus'],
                       ['Tumor_grade_Poorly_differentiated_undifferentiated','Tumor_grade_Well_moderately_differentiated'], 
                       ['Tumor_size_2_5cm','Tumor_size_5_10cm', 'Tumor_size_bigger_10cm', 'Tumor_size_smaller_2cm'],
                       ['AJCC_Stage_I', 'AJCC_Stage_II', 'AJCC_Stage_III', 'AJCC_Stage_IV'],
                       ['Mitotic_rate_bigger_5HPF', 'Mitotic_rate_smaller_5HPF'],
                       ['Surgery_Local_excision', 'Surgery_NoSurgery','Surgery_Radical_excision'], 
                       ['Regional_nodes_examined_0','Regional_nodes_examined_1to4', 'Regional_nodes_examined_bigger_4'],
                       ['Chemotherapy_No_Unknown', 'Chemotherapy_Yes'], 
                       ['Age_at_diagnosis']]
# The reference category for a variable comes from the DataPreprocessing program. 
drop_features = ['Sex_Female', 'Race_White','Marital_status_at_diagnosis_Married','Tumor_location_Antrum_Pylorus',
       'Tumor_grade_Well_moderately_differentiated', 'Tumor_size_smaller_2cm',
        'AJCC_Stage_I', 'Mitotic_rate_smaller_5HPF', 'Surgery_Local_excision', 
        'Regional_nodes_examined_0', 'Chemotherapy_Yes'] # To uniformly select the best option, which in this context means the category with the lowest risk level

def f_drop_features(col_features,drop_features):

    #To remove elements from col_features that exist in drop_features and return the updated list

    for list_i in col_features:
        for ele in list_i:
            if ele in drop_features:
                list_i.remove(ele) # remove
            else:
                pass
    return col_features

# The groups that require Univariate Analysis should be stored in the form of a list.
final_signal_features = f_drop_features(col_signal_features,drop_features)
final_signal_features''',language='python',line_numbers=True)
        st.code('''# First, perform the fitting, and then define the residual fitting and output for each feature
def each_Schoenfeld(data,feature):# "data" refers to a variable, whether it is a continuous variable or all categories of a categorical variable.
    cph = CoxPHFitter()
    cph.fit(data, 'Survival_months', 'COD')
    cph.print_summary(model="untransformed variables", decimals=3)
    
    #When drawing the forest plot below, it seems to be not very smooth, as the scales below are not uniform. Therefore, it is necessary to draw another forest plot
    fig,ax = plt.subplots(figsize=(4,6), dpi=300) #dpi=120
    cph.plot(ax=ax)# forest plot
    plt.tight_layout()
    ax.grid()# grid on
    fig.savefig(f'{feature}.pdf', dpi=200)
    
#     plt.figure(figsize=(5, 3))  
    cph.check_assumptions(data, p_value_threshold=0.05, show_plots=True)# Turn off the display because we will manually save

for feature in final_signal_features:
    feature_final_data = final_data.loc[:,feature+['Survival_months', 'COD']]
    print(feature_final_data.columns)
    each_Schoenfeld(feature_final_data,feature)

# age_final_data = final_data.loc[:,['Age_at_diagnosis','Survival_months', 'COD']]
# cph = CoxPHFitter()
# cph.fit(age_final_data, 'Survival_months', 'COD')
# cph.print_summary(model="untransformed variables", decimals=3)
# cph.check_assumptions(age_final_data, p_value_threshold=0.05, show_plots=True)''',language='python',line_numbers=True)
        st.code('''features = final_data.columns
final_two_features = [fea for fea in features if fea not in drop_features]
final_two_features''',language='python',line_numbers=True)
        st.code('''final_two_features.remove('Tumor_location_Body')
final_two_features.remove('Tumor_location_Cardia_Fundus')
final_two_features''',language='python',line_numbers=True)
        st.code('''from lifelines import CoxPHFitter
# Create a Cox regression model
cph = CoxPHFitter()
# Pass in the data in DataFrame format for df, the column name for time to duration_col, and the column name for events to event_col. By default, all covariates will be used
# You can pass in partial covariates through the parameter formula='Age+Race'.
two_data_cox = final_data.loc[:,final_two_features]
cph.fit(df=two_data_cox,duration_col='Survival_months',event_col='COD',show_progress=True)
# Print the model details
cph.print_summary()''',language='python',line_numbers=True)
        st.code('''import matplotlib.pyplot as plt
# help(cph.plot)
fig,ax = plt.subplots(figsize=(4,6), dpi=300)#dpi=120
cph.plot(ax=ax)
plt.tight_layout()
ax.grid()

fig.savefig('cox_f1.pdf', dpi=200)''',language='python',line_numbers=True)
        st.code('''# Combination of multiple features
features_final_cox_71 = ['Age_at_diagnosis', 'Sex', 'Race', 'Tumor_size', 'AJCC_Stage', 'Surgery', 'Regional_nodes_examined']#'Marital_status_at_diagnosis',
features_final_cox_72 = ['Age_at_diagnosis', 'Sex', 'Race', 'Marital_status_at_diagnosis', 'AJCC_Stage', 'Surgery', 'Regional_nodes_examined']#,'Tumor_size'
features_final_cox_73 = ['Age_at_diagnosis', 'Sex', 'Race', 'Marital_status_at_diagnosis','Tumor_size', 'Surgery', 'Regional_nodes_examined']#'AJCC_Stage',
features_final_cox_8 = ['Age_at_diagnosis', 'Sex', 'Race', 'Marital_status_at_diagnosis','Tumor_size', 'AJCC_Stage', 'Surgery', 'Regional_nodes_examined']
features_final_cox_9 = features_final_cox_8 + ['Tumor_grade']
features_final_cox_10 = features_final_cox_9 + ['Mitotic_rate']
features_final_cox_11 = features_final_cox_10 + ['Chemotherapy']
features_final_cox_all = features_final_cox_11 + ['Tumor_location']

# Tumor_grade 0.12
# Mitotic_rate 0.95
# Chemotherapy 1.2
# Tumor_location''',language='python',line_numbers=True)
        st.code('''import pandas as pd
df_css0_complete_11=pd.read_csv('df_css0_complete_11.csv') 

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from matplotlib import pyplot as plt
from lifelines import WeibullAFTFitter
import numpy as np
import pandas as pd


data_cat = pd.get_dummies(df_css0_complete_11.drop(columns=['Age_at_diagnosis','Survival_months', 'COD'],axis=1)) # String, digitization
data_cat

final_data = pd.concat([data_cat,df_css0_complete_11.loc[:,['Age_at_diagnosis','Survival_months', 'COD']]],axis=1)
final_data.columns''',language='python',line_numbers=True)
        st.code('''# 2
col_signal_features = [['Sex_Female', 'Sex_Male'], 
                       ['Race_Black', 'Race_White', 'Race_others'],
                       ['Marital_status_at_diagnosis_Married','Marital_status_at_diagnosis_Single'], 
                       ['Tumor_location_Antrum_Pylorus','Tumor_location_Body', 'Tumor_location_Cardia_Fundus'],
                       ['Tumor_grade_Poorly_differentiated_undifferentiated','Tumor_grade_Well_moderately_differentiated'], 
                       ['Tumor_size_2_5cm','Tumor_size_5_10cm', 'Tumor_size_bigger_10cm', 'Tumor_size_smaller_2cm'],
                       ['AJCC_Stage_I', 'AJCC_Stage_II', 'AJCC_Stage_III', 'AJCC_Stage_IV'],
                       ['Mitotic_rate_bigger_5HPF', 'Mitotic_rate_smaller_5HPF'],
                       ['Surgery_Local_excision', 'Surgery_NoSurgery','Surgery_Radical_excision'], 
                       ['Regional_nodes_examined_0','Regional_nodes_examined_1to4', 'Regional_nodes_examined_bigger_4'],
                       ['Chemotherapy_No_Unknown', 'Chemotherapy_Yes'], 
                       ['Age_at_diagnosis']]

drop_features = ['Sex_Female', 'Race_White','Marital_status_at_diagnosis_Married','Tumor_location_Antrum_Pylorus',
       'Tumor_grade_Well_moderately_differentiated', 'Tumor_size_smaller_2cm',
        'AJCC_Stage_I', 'Mitotic_rate_smaller_5HPF', 'Surgery_Local_excision', 
        'Regional_nodes_examined_0', 'Chemotherapy_Yes'] 

def f_drop_features(col_features,drop_features):

    for list_i in col_features:
        for ele in list_i:
            if ele in drop_features:
                list_i.remove(ele) 
            else:
                pass
    return col_features

final_signal_features = f_drop_features(col_signal_features,drop_features)
final_signal_features''',language='python',line_numbers=True)
        st.code('''for feature in final_signal_features:
    feature_final_data = final_data.loc[:,feature+['Survival_months', 'COD']]
    print(feature_final_data.columns)
    
    aft = WeibullAFTFitter()
    aft.fit(df=feature_final_data,duration_col='Survival_months',event_col='COD',show_progress=True)

    aft.print_summary()
''',language='python',line_numbers=True)
        st.code('''features = final_data.columns
final_two_features = [fea for fea in features if fea not in drop_features]
final_two_features
''',language='python',line_numbers=True)
        st.code('''
from lifelines import WeibullAFTFitter

aft = WeibullAFTFitter()
two_data_cox = final_data.loc[:,final_two_features]
aft.fit(df=two_data_cox,duration_col='Survival_months',event_col='COD',show_progress=True)

aft.print_summary()''',language='python',line_numbers=True)
        st.code('''features_final_aft_8 = ['Age_at_diagnosis', 'Sex', 'Race', 'Marital_status_at_diagnosis',
       'Tumor_location', 'Tumor_grade', 'Tumor_size', 'AJCC_Stage',
       'Mitotic_rate', 'Surgery', 'Regional_nodes_examined', 'Chemotherapy']
''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        
        st.subheader('Training And Testing Sets')
        st.code('''import pandas as pd
df_css0_complete_11=pd.read_csv('df_css0_complete_11.csv') 

drop_features = ['Sex_Male', 'Race_White','Marital_status_at_diagnosis_Married','Tumor_location_Antrum_Pylorus',
       'Tumor_grade_Well_moderately_differentiated', 'Tumor_size_smaller_2cm',
        'AJCC_Stage_I', 'Mitotic_rate_smaller_5HPF', 'Surgery_Local_excision', 
        'Regional_nodes_examined_0', 'Chemotherapy_Yes'] 


features_final_cox_71 = ['Age_at_diagnosis', 'Sex', 'Race', 'Tumor_size', 'AJCC_Stage', 'Surgery', 'Regional_nodes_examined']#'Marital_status_at_diagnosis',
features_final_cox_72 = ['Age_at_diagnosis', 'Sex', 'Race', 'Marital_status_at_diagnosis', 'AJCC_Stage', 'Surgery', 'Regional_nodes_examined']#,'Tumor_size'
features_final_cox_73 = ['Age_at_diagnosis', 'Sex', 'Race', 'Marital_status_at_diagnosis','Tumor_size', 'Surgery', 'Regional_nodes_examined']#'AJCC_Stage',
features_final_cox_8 = ['Age_at_diagnosis', 'Sex', 'Race', 'Marital_status_at_diagnosis','Tumor_size', 'AJCC_Stage', 'Surgery', 'Regional_nodes_examined']
features_final_cox_9 = features_final_cox_8 + ['Tumor_grade']
features_final_cox_10 = features_final_cox_9 + ['Mitotic_rate']
features_final_cox_11 = features_final_cox_10 + ['Chemotherapy']
features_final_cox_all = features_final_cox_11 + ['Tumor_location']


def random_split(data,features_list,random_state):
    

# data: Raw data that needs to be split 
# features_list: The feature categories of the raw data. Note that 'Survival_months' and 'COD' do not need to be included. 
# random_state: Random seed

    # Select the corresponding categorical variables from the raw data; Perform dummy variable encoding/processing.
    features_list.remove('Age_at_diagnosis')
    data_X = data.loc[:,features_list] # Excluding age from here because categorical variables require dummy variable encoding/processing.
    data_cat = pd.get_dummies(data_X)
    final_features = [fea for fea in data_cat.columns if fea not in drop_features]
    final_data = pd.concat([data_cat.loc[:,final_features],data.loc[:,['Age_at_diagnosis','Survival_months', 'COD']]],axis=1)
    
    # 3. Split the data into a 7:3 training set and test set, and return them. Note that we do not split the data directly here, but instead add a column labeled "label" to the dataset later.
    train_data = final_data.sample(frac=0.7, random_state=random_state) 
    test_data = final_data.drop(train_data.index)
    train_data['label']='train'
    test_data['label']='test'
    table1_css = pd.concat([train_data,test_data],axis=0)
    
    return table1_css
    


cox_8 = random_split(df_css0_complete_11,features_final_cox_8,RANDOM_STATE) 

''',language='python',line_numbers=True)
        st.code('''from lifelines import KaplanMeierFitter

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from matplotlib import pyplot as plt
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd

# ax = plt.subplot(111)
fig, ax = plt.subplots()  
kmf = KaplanMeierFitter()

# plt.tight_layout()
# plt.savefig('xx.pdf',dpi=300)

for name, grouped_df in cox_8.groupby('label'): 
    kmf.fit(grouped_df["Survival_months"], grouped_df["COD"], label=name)
    kmf.plot_survival_function(ax=ax) 
#     plt.tight_layout()
#     plt.savefig('km154.pdf',dpi=300)

plt.grid(True)
plt.tight_layout()
plt.savefig('km1.pdf', dpi=300)
plt.show()''',language='python',line_numbers=True)
        st.code('''from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


result = multivariate_logrank_test(cox_8['Survival_months'], cox_8['label'], cox_8['COD'])
result.test_statistic
result.p_value
result.print_summary()


# If the p-value of the Log-rank test is less than this level, then it can be considered that there is a statistically significant difference between the survival curves of the two categories.
# If the p-value is greater than 0.05, then it indicates that there is no significant difference between the training and test sets.
''',language='python',line_numbers=True)
        st.code('''import pandas as pd
df_css0_complete_11=pd.read_csv('df_css0_complete_11.csv') 

# Try it without performing dummy processing, and directly use the chi-square test and U test instead.
from tableone import TableOne

train_data = df_css0_complete_11.sample(frac=0.7, random_state=RANDOM_STATE) 

test_data = df_css0_complete_11.drop(train_data.index)

train_data['label']='train'
test_data['label']='test'

table1_css = pd.concat([train_data,test_data],axis=0)

table1_css.columns''',language='python',line_numbers=True)
        st.code('''# All the relevant features, including the columns used for grouping (groupby).
columns =['Age_at_diagnosis', 'Sex', 'Race', 'Marital_status_at_diagnosis',
       'Tumor_location', 'Tumor_grade', 'Tumor_size', 'AJCC_Stage',
       'Mitotic_rate', 'Surgery', 'Regional_nodes_examined', 'Chemotherapy','label']
# Categorical variable
categorical = ['Sex', 'Race', 'Marital_status_at_diagnosis',
       'Tumor_location', 'Tumor_grade', 'Tumor_size', 'AJCC_Stage',
       'Mitotic_rate', 'Surgery', 'Regional_nodes_examined', 'Chemotherapy']
# For continuous variables with non-normal distribution, if not specified, it is estimated that the t-test will be automatically performed.
nonnormal = ['Age_at_diagnosis']
# Grouping, where the label is selected to provide the division of training and testing sets.
groupby = ['label']


# create grouped_table with p values
table3 = TableOne(table1_css, columns, categorical, groupby, nonnormal, pval = True, htest_name=True)
# view first 10 rows of tableone
table3
# Save to Excel
table3.to_excel('seer2_css.xlsx')

table3''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        
        st.subheader('Model Training And Evaluation')
        st.code('''''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        st.subheader('Model Saving And Deployment')
        st.code('''''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)
        st.code('''''',language='python',line_numbers=True)

        

##########################################################    3    #############################################################


elif selected_option == 'radiomics Comming soon':

    st.write(f"comming soon...")

# elif selected_option == 'Pie Chart':

#     st.write(f"comming soon")

# else:
#     st.write('Invalid selection')
