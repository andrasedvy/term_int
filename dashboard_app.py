# dashboard_app.py – Streamlit dashboard karbantartási javaslatokkal

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from motor_model_simple import DCMotor
from fault_diagnosis import FaultDiagnosis

st.set_page_config(page_title="DC Motor Diagnosztika", layout="wide")

st.title("Ipari DC Motor Diagnosztika – Streamlit")

# --- Paraméterek ---
st.sidebar.header("Paraméterek")
s_url_mertek = st.sidebar.slider("Súrlódás mértéke", 0.0, 1.0, 0.0, 0.05)
s_url_ido = st.sidebar.slider("Súrlódás kezdete (s)", 0.0, 20.0, 5.0, 0.5)
rz_mertek = st.sidebar.slider("Zárlat mértéke", 0.0, 1.0, 0.0, 0.05)
rz_ido = st.sidebar.slider("Zárlat kezdete (s)", 0.0, 20.0, 10.0, 0.5)
Tamb = st.sidebar.slider("Környezeti hőmérséklet (°C)", 20, 60, 25)
omega_ref = st.sidebar.slider("Referencia fordulatszám [rad/s]", 0.5, 2.0, 1.2, 0.1)

# --- Szimuláció ---
def pwm_fn(t, omega, omega_ref):
    return 0.3 + 1.5 * (omega_ref - omega)

motor = DCMotor(T_end=20.0, debug=False)
t_vals, I_vals, omega_vals, T_vals = motor.simulate(
    pwm_fn=pwm_fn,
    omega_ref=omega_ref,
    s_url_ido=s_url_ido if s_url_mertek > 0 else None,
    s_url_mertek=s_url_mertek,
    rz_ido=rz_ido if rz_mertek > 0 else None,
    rz_mertek=rz_mertek,
    Tamb=Tamb
)

# --- Diagnózis és javaslat ---
diagnoser = FaultDiagnosis()
labels = diagnoser.analyze(t_vals, I_vals, omega_vals, T_vals)
actions = diagnoser.suggest_actions(labels)

# --- Megjelenítés ---
label_names = ["Súrlódás", "Áramhiba", "Túlmelegedés"]
colors = ['orange', 'red', 'purple']

st.subheader("Folyamatváltozók")
cols = st.columns(3)
with cols[0]:
    fig1, ax1 = plt.subplots(figsize=(4, 2.5))
    ax1.plot(t_vals, I_vals); ax1.set_title("Áram [A]"); ax1.grid(True)
    st.pyplot(fig1)
with cols[1]:
    fig2, ax2 = plt.subplots(figsize=(4, 2.5))
    ax2.plot(t_vals, omega_vals, color='orange'); ax2.set_title("Fordulatszám [rad/s]"); ax2.grid(True)
    st.pyplot(fig2)
with cols[2]:
    fig3, ax3 = plt.subplots(figsize=(4, 2.5))
    ax3.plot(t_vals, T_vals, color='red'); ax3.set_title("Hőmérséklet [°C]"); ax3.grid(True)
    st.pyplot(fig3)

# --- Diagnosztika ---
st.subheader("Prediktált hibák")
fig4, ax4 = plt.subplots(figsize=(10, 2.5))

for i in range(3):
    idxs = np.where(labels[:, i] == 1)[0]
    ax4.plot(t_vals[idxs], np.ones_like(idxs) * (i + 1), '.', label=f"{label_names[i]} ({len(idxs)})", color=colors[i], markersize=2)

ax4.set_yticks([1, 2, 3])
ax4.set_yticklabels(label_names)
ax4.set_xlabel("Idő [s]")
ax4.set_title("Diagnosztikai események")
ax4.grid(True); ax4.legend()
st.pyplot(fig4)

# --- Karbantartási javaslatok ---
st.subheader("Karbantartási javaslatok")
for action in actions:
    st.write("- " + action)
