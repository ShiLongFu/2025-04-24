import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('END_XGBoost_model.pkl')

# 定义特征的选项
SDH_options = {
    0: 'No (0)',
    1: 'Yes (1)'
}

Severe_tSAH_options = {
    0: 'No (0)',
    1: 'Yes (1)'
}

# Define feature names
feature_names = ["Hemoglobin", "Fibrinogen", "SDH", "Severe tSAH"]

# Streamlit的用户界面
st.title("Early Neurological Deterioration (END) Predictor")

# Hemoglobin: 数值输入
Hemoglobin = st.number_input("Hemoglobin (g/L):", min_value=0, max_value=200, value=120)

# Fibrinogen: 数值输入
Fibrinogen = st.number_input("Fibrinogen (g/L):", min_value=0.01, max_value=20.00, value=1.00)

# contusion: 分类选择
SDH = st.selectbox("Subdural hemorrhage (SDH):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# tSAH: 分类选择
Severe_tSAH = st.selectbox("Severe traumatic subarachnoid hemorrhage (tSAH) (Morris-Marshall Grade 3 or 4):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# 处理输入并进行预测
feature_values = [Hemoglobin, Fibrinogen, SDH, Severe_tSAH]
features = np.array([feature_values])

if st.button("Predict"):
    # 预测概率（修改这里）
    predicted_proba = model.predict_proba(features)[0]
    probability_positive = predicted_proba[1] * 100  # 直接提取阳性概率
  
    # 显示结果（更新变量名）
    text = f"Based on feature values, predicted probability of END is {probability_positive:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.tif", bbox_inches='tight', dpi=300)
    st.image("prediction_text.tif")

    # SHAP部分（保持不变，但解释类别可能需要调整）
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    
    # 使用类别1的SHAP解释（如果需要展示阳性的解释）
   # 计算SHAP值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
   # 使用类别1的SHAP值
    shap.force_plot(explainer.expected_value[1], 
                shap_values[1][0],  # 注意双索引
                pd.DataFrame([feature_values], columns=feature_names),
                matplotlib=True)
    
    plt.savefig("shap_force_plot.tif", bbox_inches='tight', dpi=1200)
  
    st.image("shap_force_plot.tif")
# 运行Streamlit命令生成网页应用

