# motor_model_simple.py – ipari digitális iker, egyszerűsített és valósághű

import numpy as np

class SimpleDCMotor:
    """
    Egyszerűsített, de iparilag hűséges DC motor modell PWM vezérléssel.
    Támogatott hibák: súrlódásnövekedés, zárlat, túlmelegedés.
    """

    def __init__(self, dt=0.001, T_end=20.0, debug=False):
        self.dt = dt
        self.T_end = T_end
        self.time = np.arange(0, T_end, dt)
        self.debug = debug

        # Fizikai paraméterek (a DCmotor2.pptx alapján)
        self.R = 2.0          # Ellenállás [Ohm]
        self.L = 0.5          # Induktivitás [H]
        self.Kt = 0.1         # Nyomatéki állandó [Nm/A]
        self.Ke = 0.1         # Ellenfeszültségi állandó [V/(rad/s)]
        self.B = 0.2          # Súrlódás [Nms/rad]
        self.J = 0.02         # Tehetetlenség [kg*m^2]
        self.Cth = 100.0      # Hőkapacitás [J/°C]
        self.Vmax = 24.0      # Tápfeszültség maximum [V]

    def simulate(self,
                 pwm_fn,
                 omega_ref=1.0,
                 s_url_ido=None,
                 s_url_mertek=0.0,
                 rz_ido=None,
                 rz_mertek=0.0,
                 Tamb=25.0):

        n = len(self.time)
        omega = np.zeros(n)
        current = np.zeros(n)
        temperature = np.zeros(n)
        I = 0.0
        T_motor = Tamb

        for i in range(1, n):
            t = self.time[i]

            # Hibák időalapú bevezetése
            B_eff = self.B + s_url_mertek if s_url_ido and t >= s_url_ido else self.B
            R_eff = self.R * (1 - rz_mertek) if rz_ido and t >= rz_ido else self.R

            # PWM jel alapján bemeneti feszültség
            duty = np.clip(pwm_fn(t, omega[i-1], omega_ref), 0.2, 1.0)
            V_in = self.Vmax * duty

            # Áram differenciálegyenlet: dI/dt = (V - Ke*ω - R*I) / L
            dI_dt = (V_in - self.Ke * omega[i-1] - R_eff * I) / self.L
            I += dI_dt * self.dt
            current[i] = I

            # Szögsebesség: dω/dt = (Kt*I - B*ω) / J
            domega = (self.Kt * I - B_eff * omega[i-1]) / self.J
            omega[i] = max(0, omega[i-1] + domega * self.dt)

            # Hőmérséklet: ΔT = I^2 * R * dt / Cth
            T_motor += (I ** 2 * R_eff * self.dt) / self.Cth
            temperature[i] = T_motor

            if self.debug and i % int(1.0 / self.dt) == 0:
                print(f"t={t:.2f}s | I={I:.2f}A | ω={omega[i]:.2f}rad/s | T={T_motor:.2f}°C")

        return self.time, current, omega, temperature


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def pwm_control(t, omega, omega_ref):
        error = omega_ref - omega
        return 0.3 + 1.5 * error

    motor = SimpleDCMotor(T_end=20.0, debug=True)
    t, I, w, T = motor.simulate(
        pwm_fn=pwm_control,
        omega_ref=1.2,
        s_url_ido=5.0,
        s_url_mertek=0.5,
        rz_ido=10.0,
        rz_mertek=0.8,
        Tamb=22.0
    )

    plt.figure(figsize=(10, 6))
    plt.subplot(3,1,1); plt.plot(t, I); plt.ylabel("Áram [A]")
    plt.subplot(3,1,2); plt.plot(t, w); plt.ylabel("Fordulatszám [rad/s]")
    plt.subplot(3,1,3); plt.plot(t, T); plt.ylabel("Hőmérséklet [°C]"); plt.xlabel("Idő [s]")
    plt.tight_layout(); plt.show()
