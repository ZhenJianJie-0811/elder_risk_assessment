# elder_risk_assessment
社工家訪長者個案風險評估系統（Streamlit + XGBoost）

本專案提供一個以 Streamlit 建構的本地端預測系統，協助社工評估個案風險等級。
系統使用 XGBoost 分類模型，透過 20 個個案指標產生風險預測與信心度。

專案內容
app.py                     # 主應用程式（Streamlit）
social_work_model.json     # 訓練後的 XGBoost 模型
feature_names.pkl          # 模型對應的特徵名稱
requirements.txt           # 套件需求檔
README.md                  # 專案說明
.gitignore                 # Git 忽略規則

安裝與環境建置
1. 安裝 Python（≥3.10）
可從官方下載：
https://www.python.org/downloads/
請勾選 Add Python to PATH。

2. 安裝所需套件
進入專案資料夾後執行：
pip install -r requirements.txt

3. 執行系統
streamlit run app.py
執行後會看到：
Local URL: http://localhost:8501
(localhost 需輸入當前網路的ipv4 地址)


系統功能說明
1. 表單輸入 20 個個案特徵
每個欄位都具有中文名稱與原始變數代號。

2. 模型推論
系統會輸出：
風險等級（1、2、3）
模型信心度（%）
三分類機率分布圖

4. 輸出建議
依照不同風險級別提供對應的建議文本。

專案檔案結構與說明
project-root/
│
├── app.py
│   ├── Streamlit UI 建立
│   ├── 載入 XGBoost 模型
│   ├── 載入特徵名稱
│   ├── 表單生成與數據收集
│   └── 預測與結果顯示
│
├── social_work_model.json
│   └── 訓練完成的 XGBoost 模型檔
│
├── feature_names.pkl
│   └── 模型訓練時的特徵名稱（預測時需一致）
│
├── requirements.txt
│   └── 專案所需 Python 套件
│
├── .gitignore
│   └── Git 忽略規則（避免上傳暫存檔案）
│
└── README.md
    └── 專案說明與使用教學
