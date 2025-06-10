import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('END_XGBoost_model.pkl')

# 定义特征名称
feature_names = ["Hemoglobin", "Fibrinogen", "SDH", "Severe tSAH"]

# Streamlit界面设置
st.title("Early Neurological Deterioration (END) Predictor")

# 输入控件
Hemoglobin = st.number_input("Hemoglobin (g/L):", min_value=0, max_value=200, value=120)
Fibrinogen = st.number_input("Fibrinogen (g/L):", min_value=0.01, max_value=20.00, value=1.00)
SDH = st.selectbox("Subdural hemorrhage (SDH):", options=[0,1], 
                  format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Severe_tSAH = st.selectbox("Severe traumatic subarachnoid hemorrhage (tSAH):", options=[0,1],
                          format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

if st.button("Predict"):
    # 准备数据
    features = pd.DataFrame([[Hemoglobin, Fibrinogen, SDH, Severe_tSAH]], 
                           columns=feature_names)
    
    # 预测概率
    proba = model.predict_proba(features)[0][1] * 100
    
    # 结果显示（优化为直接显示）
    st.markdown(f"## Prediction Result")
    st.success(f"Predicted probability of END is **{proba:.2f}%**")

    # SHAP解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # 动态生成SHAP图
    st.markdown("## Feature Impact Analysis")
    fig, ax = plt.subplots(figsize=(12,4))
    shap.force_plot(explainer.expected_value[1],  # 使用正类期望值
                   shap_values[1], 
                   features,
                   matplotlib=True,
                   show=False)
    st.pyplot(fig)
    plt.close()
