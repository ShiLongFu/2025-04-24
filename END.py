import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('END_XGBoost_model.pkl')

# 定义特征的选项
feature_names = ["Hemoglobin", "Fibrinogen", "SDH", "Severe tSAH"]

# Streamlit的用户界面
st.title("Early Neurological Deterioration (END) Predictor")

# 输入组件
Hemoglobin = st.number_input("Hemoglobin (g/L):", min_value=0, max_value=200, value=120)
Fibrinogen = st.number_input("Fibrinogen (g/L):", min_value=0.01, max_value=20.00, value=1.00)
SDH = st.selectbox("Subdural hemorrhage (SDH):", options=[0, 1], 
                  format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Severe_tSAH = st.selectbox("Severe traumatic subarachnoid hemorrhage (tSAH) (Morris-Marshall Grade 3 or 4):", 
                          options=[0, 1], 
                          format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# 处理输入并进行预测
feature_values = [Hemoglobin, Fibrinogen, SDH, Severe_tSAH]
features_df = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
    # 预测概率
    predicted_proba = model.predict_proba(features_df)[0]
    probability_positive = predicted_proba[1] * 100

    # 显示预测结果
    st.markdown(f"<h3 style='text-align: center;'>Predicted probability of END: {probability_positive:.2f}%</h3>", 
                unsafe_allow_html=True)

    # SHAP解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(features_df)
    
    # 创建SHAP force plot
    plt.figure()
    shap_plot = shap.force_plot(
        base_value=explainer.expected_value[1],  # 使用正类期望值
        shap_values=shap_values.values[:, :, 1],  # 选择正类SHAP值
        features=features_df,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=150)
    st.image("shap_force_plot.png")

# 运行: streamlit run your_script.py

