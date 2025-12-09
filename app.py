import streamlit as st
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
import os

# 設定頁面
st.set_page_config(page_title="社工個案風險評估", page_icon="📋")
st.title("📋 社工個案風險評估系統")
st.markdown("---")


DISPLAY_NAMES = {
    'C2.4':'1.最近 1 個月在生活上遭遇的困難-最近記憶力不好(C2.4, importance: 0.137)',       
    'C10.1.2':'2.接受其他服務的意願-安裝緊急救援裝置(C10.1.2, importance:  0.0837)',     
    'C1.2':'3.不可以自我照顧-使用器具(例如輪椅、拐杖)就可以自行移動(C1.2, importance: 0.787)',       
    'C10.1.1':'4.接受其他服務的意願-志工關懷訪視(C10.1.1, importance: 0.065)',      
    'C4.2':'5.與鄰居聯繫互動，大約情形是？(C4.2, importance: 0.060)',        
    'S1.2':'6.疾病-眼部疾病(S1.2, importance: 0.047)',         
    'C2.3':'7.最近 1 個月在生活上遭遇的困難-外出交通不方便(C2.3, importance: 0.047)',        
    'C8.1.7':'8.最近1個月感到鬱悶的事情-其他(C8.1.7, importance: 0.045)',  
    'C1.5':'9.不可以自我照顧-其他(C1.5, importance: 0.044)',         
    'C8.1.4':'10.最近1個月感到鬱悶的事情-子女、孫子女問題(C8.1.4, importance: 0.044)',       
    'S1.4':'11.疾病-糖尿病(S1.4, importance: 0.042)',        
    'S1.3':'12.疾病-心臟病(S1.3, importance: 0.041)',         
    'S1.9':'13.疾病-高血壓(S1.9, importance: 0.040)',         
    'C2.2':'14.最近 1 個月在生活上遭遇的困難-無人可協助就醫(C2.2, importance: 0.039)',         
    'S1.12':'15.疾病-其他(S1.12, importance: 0.037)',       
    'S1.5':'16.疾病-骨與關節疾病(S1.5, importance: 0.035)',        
    'C4.1':'17.與親友聯繫互動，大約情形是？(C4.1, importance: 0.034)',        
    'C1.3':'18.不可以自我照顧-身上有異味(C1.3, importance: 0.032)',         
    'C10.1.4':'19.接受其他服務的意願-縣市政府轉介服務-長照服務(C10.1.4, importance: 0.029)',     
    'B12.3.a_C2':'20.歿兒子幾人_2組(B12.3.a_C2, importance: 0.028)'   
}

# ==========================================
# 1. 載入模型與設定
# ==========================================
@st.cache_resource
def load_model_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "social_work_model.json")
    features_path = os.path.join(current_dir, "feature_names.pkl")
    
    try:
        if not os.path.exists(model_path): return None, None, f"找不到: {model_path}"
        if not os.path.exists(features_path): return None, None, f"找不到: {features_path}"

        model = xgb.XGBClassifier()
        model.load_model(model_path)
        feature_names = joblib.load(features_path)
        return model, feature_names, "Success"
    except Exception as e:
        return None, None, str(e)

model, feature_names, status_msg = load_model_data()

if status_msg != "Success":
    st.error(f"⚠️ 錯誤: {status_msg}")
    st.stop()

# ==========================================
# 2. 建立輸入表單 (自動翻譯)
# ==========================================
st.subheader("📝 請輸入個案指標")
inputs = {}

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    # 遍歷原本的 20 個特徵代號
    for i, code in enumerate(feature_names):
        
        # 1. 嘗試從字典抓取中文名稱，抓不到就用原代號
        label_text = DISPLAY_NAMES.get(code, code)
        
        # 2. 顯示輸入框
        with (col1 if i % 2 == 0 else col2):
            val = st.number_input(
                label=label_text,  # 這裡顯示中文
                value=0, 
                step=1, 
                format="%d",
                help=f"原始代號: {code}" # 滑鼠移過去會顯示代號，方便除錯
            )
            
            # 3. 【關鍵】存回 inputs 時，一定要用「原始代號」當 Key
            inputs[code] = val
    
    submit = st.form_submit_button("🚀 開始分析", type="primary")

# ==========================================
# 3. 推論與結果
# ==========================================
if submit:
    # 轉成 DataFrame (這裡的欄位名稱會是 C2.4 等代號，模型才看得懂)
    input_df = pd.DataFrame([inputs])
    
    # 預測
    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    
    risk_level = pred_class + 1 
    confidence = pred_proba[pred_class] * 100

    st.markdown("---")
    st.subheader("📊 分析結果")
    
    if risk_level == 1:
        st.success(f"✅ 評估等級：1 (低風險)")
        st.metric("模型信心度", f"{confidence:.1f}%")
        st.info("建議：維持常規追蹤即可。")
    elif risk_level == 2:
        st.warning(f"⚠️ 評估等級：2 (中風險)")
        st.metric("模型信心度", f"{confidence:.1f}%")
        st.markdown("**建議：需增加訪視頻率，密切注意指標變化。**")
    else:
        st.error(f"🚨 評估等級：3 (高風險)")
        st.metric("模型信心度", f"{confidence:.1f}%")
        st.markdown("### 建議：立即介入處理！")
        
    with st.expander("查看詳細機率分佈"):
        st.bar_chart(pd.DataFrame(pred_proba, index=["Level 1", "Level 2", "Level 3"], columns=["機率"]))