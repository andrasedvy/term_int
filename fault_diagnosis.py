# fault_diagnosis.py – túlmelegedés logika + karbantartási javaslat + dashboard támogatás

import numpy as np

class FaultDiagnosis:
   
    
    def __init__(self,
                 friction_thresh=0.2,
                 current_thresh=6.5,
                 temp_thresh=40.0,
                 temp_rate_thresh=1.5):
        self.friction_thresh = friction_thresh
        self.current_thresh = current_thresh
        self.temp_thresh = temp_thresh
        self.temp_rate_thresh = temp_rate_thresh

    def analyze(self, t, current, omega, temperature):
        n = len(t)
        labels = np.zeros((n, 3), dtype=int)

        omega_ref = np.median(omega[:int(n * 0.1)])
        current_baseline = np.median(current[:int(n * 0.1)])
        temp_baseline = np.median(temperature[:int(n * 0.1)])

        for i in range(n):
            dT = temperature[i] - temperature[i - 3] if i >= 3 else 0

            # Súrlódás
            if omega[i] < (1 - self.friction_thresh) * omega_ref and current[i] > 1.2 * current_baseline:
                labels[i, 0] = 1

            # Áramhiba
            if current[i] > self.current_thresh:
                labels[i, 1] = 1

            # Túlmelegedés: érzékenyített logika kombinált feltételekkel
            is_temp_high = temperature[i] > self.temp_thresh
            is_fast_rise = dT > self.temp_rate_thresh and t[i] > 2.0
            is_delta = temperature[i] - temp_baseline > 10
            is_low_rpm = omega[i] < 0.75 * omega_ref
            is_high_current = current[i] > 1.2 * current_baseline

            if (is_temp_high or is_fast_rise or is_delta) and is_low_rpm and is_high_current:
                labels[i, 2] = 1

        return labels

    def suggest_actions(self, labels):
        summary = labels.sum(axis=0)
        actions = []

        if summary[0] > 0:
            actions.append("Súrlódási hiba: Ellenőrizze a csapágyakat, kenést, mechanikai ellenállásokat.")
        if summary[1] > 0:
            actions.append("Áramhiba: Vizsgálja a tekercseket, zárlatot, áramkör túlterhelését.")
        if summary[2] > 0:
            actions.append("Túlmelegedés: Tisztítsa a hűtőrendszert, csökkentse a terhelést, állítsa le a rendszert.")

        if not actions:
            actions.append(" Nincs észlelt hiba. A rendszer jelenleg normál állapotban működik.")

        return actions


if __name__ == "__main__":
    from motor_model_simple import DCMotor
    import matplotlib.pyplot as plt
    import streamlit as st

    def pwm_control(t, omega, omega_ref):
        return 0.3 + 1.5 * (omega_ref - omega)

    motor = DCMotor(T_end=20.0, debug=False)
    t, I, w, T = motor.simulate(
        pwm_fn=pwm_control,
        omega_ref=2.0,
        s_url_ido=5.0,
        s_url_mertek=1.0,
        rz_ido=10.0,
        rz_mertek=0.0,
        Tamb=25.0
    )

    diagnoser = FaultDiagnosis()
    labels = diagnoser.analyze(t, I, w, T)
    actions = diagnoser.suggest_actions(labels)

    plt.figure(figsize=(10, 3))
    plt.plot(t, labels[:, 0], label="Súrlódás")
    plt.plot(t, labels[:, 1], label="Áramhiba")
    plt.plot(t, labels[:, 2], label="Túlmelegedés")
    plt.legend(); plt.title("Diagnosztika és karbantartási javaslat"); plt.xlabel("Idő [s]")
    plt.tight_layout(); plt.show()

    st.markdown("### 🛠 Karbantartási javaslatok:")
    for action in actions:
        st.write("- " + action)
