import streamlit as st
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import os

from LSTMS2S_NH4 import Config, Seq2SeqLSTM, process_data, WaterSeqDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(page_title="Wastewater Treatment Dashboard", page_icon="💧", layout="wide")

# Custom CSS for UI
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-warning {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #ff4d4d;
    }
    .metric-good {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #4dff4d;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Tasks 2.1 & 2.2: Load Models
# ----------------------------
@st.cache_resource
def load_resources():
    # LSTM
    train_data, test_data, scaler_X, scaler_y, feature_cols = process_data(Config)
    if train_data is None:
        return None, None, None, None, None, None

    model = Seq2SeqLSTM(
        input_dim=len(feature_cols),
        hidden_dim=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        output_dim=len(Config.TARGET_COLS),
        pred_len=Config.PRED_LEN,
        dropout=Config.DROPOUT
    ).to(device)

    model_path = os.path.join('mode', 'best_model_NH4.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        st.error("LSTM Model not found.")

    # PPO Agent
    agent_path = 'ppo_agent_16h.zip'
    agent = None
    if os.path.exists(agent_path):
        agent = PPO.load(agent_path, device=device)
    else:
        st.error("PPO Agent not found.")

    return model, agent, scaler_X, scaler_y, feature_cols, test_data

lstm_model, ppo_agent, scaler_X, scaler_y, feature_cols, test_data = load_resources()

# ----------------------------
# Task 2.3 & 2.4: Helpers
# ----------------------------
def predict_effluent_nh4(state_sequence_scaled):
    x_tensor = torch.tensor(state_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_seq = lstm_model(x_tensor).cpu().numpy()
    last_point_pred = pred_seq[:, -1, :] 
    pred_real = scaler_y.inverse_transform(last_point_pred)
    return pred_real[0, 0]

def get_rl_recommendation(state_sequence_scaled):
    # PPO agent expected shape is (36, 16)
    obs = state_sequence_scaled
    action, _ = ppo_agent.predict(obs, deterministic=True)
    return action[0], action[1]  # scaled Ri, Re [0, 1]

if test_data is None:
    st.stop()

# Get latest sequence from test_data
X_test, y_test = test_data
latest_sequence = X_test[-Config.SEQ_LEN:].copy()  # Shape: (SEQ_LEN, input_dim)

idx_Ri = feature_cols.index('Ri')
idx_Re = feature_cols.index('Re')

# Current Parameters
current_Ri_scaled = latest_sequence[-1, idx_Ri]
current_Re_scaled = latest_sequence[-1, idx_Re]

# Unscale for display
dummy_feature = np.zeros((1, len(feature_cols)))
dummy_feature[0, idx_Ri] = current_Ri_scaled
dummy_feature[0, idx_Re] = current_Re_scaled
dummy_real = scaler_X.inverse_transform(dummy_feature)
current_Ri_real = dummy_real[0, idx_Ri]
current_Re_real = dummy_real[0, idx_Re]

# Fetch Date from rawdata.csv
df_raw = pd.read_csv('rawdata.csv')
latest_date_str = str(df_raw['Date'].iloc[-1]) if 'Date' in df_raw.columns else "Unknown Time"

# ----------------------------
# Layout
# ----------------------------
st.title("💧 Wastewater Treatment Plant: RL Operator Dashboard")
st.markdown("---")

# Task 3.2: Current Plant Status
st.header(f"1. Current Plant Status (Latest Reading: {latest_date_str})")

# Unscale all features for the latest reading
dummy_real_all = scaler_X.inverse_transform(latest_sequence[-1].reshape(1, -1))[0]
def get_val(feat):
    if feat in feature_cols:
        return f"{dummy_real_all[feature_cols.index(feat)]:.2f}"
    return "N/A"

st.subheader("⚙️ 總進水與操作參數 (Influent & Operations)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ri (內迴流)", get_val('Ri'))
c2.metric("Re (外迴流)", get_val('Re'))
c3.metric("C/N 比", get_val('C/N'))

st.subheader("🧪 缺氧槽 (Anoxic Reactor)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("pH", get_val('PH'))
c2.metric("EC", get_val('EC'))
c3.metric("ORP1", get_val('ORP1'))

st.subheader("🫧 好氧槽 (Aerobic Reactor)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("DO (溶氧)", get_val('DO'))
c2.metric("ORP2", get_val('ORP2'))
c3.metric("SS2", get_val('SS2'))
c4.metric("COD (好氧槽)", get_val('COD'))

st.subheader("💧 出水與沉澱池 (Effluent & Settler)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("NH4 (氨氮出水)", get_val('NH4'))
c2.metric("COD (出水)", get_val('CODeff'))
c3.metric("NO3 (硝酸鹽)", get_val('NO3'))
c4.metric("SS1", get_val('SS1'))

st.markdown("---")
st.header("2. 16-Hour Prediction Scenarios")

# Scenario 1: Current Operation
# Task 4.1
pred_nh4_current = predict_effluent_nh4(latest_sequence)

# Scenario 2: RL Recommended Operation
# Task 4.2
rl_Ri_scaled, rl_Re_scaled = get_rl_recommendation(latest_sequence)

# Update sequence with RL action for prediction
rl_sequence = latest_sequence.copy()
rl_sequence[-1, idx_Ri] = rl_Ri_scaled
rl_sequence[-1, idx_Re] = rl_Re_scaled
pred_nh4_rl = predict_effluent_nh4(rl_sequence)

# Unscale RL actions for display
dummy_rl = np.zeros((1, len(feature_cols)))
dummy_rl[0, idx_Ri] = rl_Ri_scaled
dummy_rl[0, idx_Re] = rl_Re_scaled
dummy_rl_real = scaler_X.inverse_transform(dummy_rl)
rl_Ri_real = dummy_rl_real[0, idx_Ri]
rl_Re_real = dummy_rl_real[0, idx_Re]

c1, c2 = st.columns(2)

# Task 4.3 Visual cues
def get_metric_class(val):
    return "metric-warning" if val >= 15.0 else "metric-good"

with c1:
    st.subheader("Keep Current Operation")
    st.write(f"**Ri:** {current_Ri_real:.2f} | **Re:** {current_Re_real:.2f}")
    css_class = get_metric_class(pred_nh4_current)
    st.markdown(f'<div class="{css_class}"><h3>Predicted NH4 (16h)</h3><h1>{pred_nh4_current:.2f} mg/L</h1></div>', unsafe_allow_html=True)

with c2:
    st.subheader("RL Recommended Operation")
    st.write(f"**Ri:** {rl_Ri_real:.2f} | **Re:** {rl_Re_real:.2f}")
    css_class = get_metric_class(pred_nh4_rl)
    st.markdown(f'<div class="{css_class}"><h3>Predicted NH4 (16h)</h3><h1>{pred_nh4_rl:.2f} mg/L</h1></div>', unsafe_allow_html=True)

st.info("💡 **提示：** 目前所顯示的數據來自於資料集的最後一個觀測時間點。另外，因為 LSTM 模型是基於過去 36 小時的歷史序列進行預測，單單改變『當下這 1 小時』的 Ri 與 Re 參數，對於 16 小時後的出水預測影響可能較為平緩，或因為建議參數與現有參數相近，導致預測結果非常接近或相同。")
