import numpy as np
import matplotlib.pyplot as plt
# ======== EDITAR AQUI =================================
g = 9.8        # aceleração da gravidade (m/s^2)
m = 0.172       # massa (kg)                            0.172 cebola e 0.05 papel
k = 0.12       # coef. de resistência linear (s^-1)     0.12 cebola e 1.28 papel
c = 0.35       # coef. de resistência quadrática (kg/m) 0.35 cebola e 2.94 papel
x0 = 0.0       # posição inicial (m)
v0 = 0.0       # velocidade inicial (m/s)
T = 10.0       # tempo final (s)
h = 0.01       # passo de integração (s)
# =======================================================
# -------------------------------------------------------
# 1) Queda com resistência LINEAR (solução analítica e Euler)
# -------------------------------------------------------
def v_linear_analitica(t, m, k, g):
    return (m*g/k)*(1 - np.exp(-k*t/m))

def x_linear_analitica(t, m, k, g, x0=0):
    return x0 + (m*g/k)*t - (m**2*g/k**2)*(1 - np.exp(-k*t/m))

def euler_linear(T, h, m, k, g, v0=0):
    n = int(T/h)
    t = np.linspace(0, T, n+1)
    v = np.zeros_like(t)
    v[0] = v0
    for i in range(n):
        v[i+1] = v[i] + h*(g - (k/m)*v[i])
    return t, v

# -------------------------------------------------------
# 2) Queda com resistência QUADRÁTICA
# -------------------------------------------------------
def v_quadratica_analitica(t, m, c, g, v0=0):
    if c == 0:
        return v0 + g*t
    vT = np.sqrt(m*g/c)
    z0 = np.clip(v0/vT, -0.999999, 0.999999)  # tem que estar entre -1 e 1 se nao sai do dominio da archtanh
    atanh_z0 = 0.5*np.log((1+z0)/(1-z0))
    return vT*np.tanh(np.sqrt(g*c/m)*t + atanh_z0)

def euler_quadratico(T, h, m, c, g, v0=0):
    n = int(T/h)
    t = np.linspace(0, T, n+1)
    v = np.zeros_like(t)
    v[0] = v0
    for i in range(n):
        v[i+1] = v[i] + h*(g - (c/m)*v[i]**2)
    return t, v

# -----------------------------
# Grade (t, v) para o campo
# -----------------------------
t_vals = np.linspace(0, 8, 21)
v_vals = np.linspace(-5, 30, 23)
Tg, Vg = np.meshgrid(t_vals, v_vals)

# -----------------------------
# Campo de direções — dv/dt = g - (k/m) v
# -----------------------------
dVdt = g - (k/m)*Vg
dTdt = np.ones_like(dVdt)  # componente horizontal (dt/dt = 1)

# Normalização (só para visual)
mag = np.sqrt(dTdt**2 + dVdt**2)
U = dTdt / mag
W = dVdt / mag

# Velocidade terminal
vt = m*g/k

plt.figure()
plt.quiver(Tg, Vg, U, W, angles='xy', pivot='mid', alpha=0.8)
plt.axhline(vt, linestyle=':', label=fr'$v_t = {vt:.2f}\ \mathrm{{m/s}}$')
plt.xlim(t_vals.min(), t_vals.max())
plt.ylim(v_vals.min(), v_vals.max())
plt.xlabel('t (s)')
plt.ylabel('v (m/s)')
plt.title('Campo de direções — Bola')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Execuções e Gráficos
# -------------------------------------------------------

t = np.arange(0, T+h, h)

# ---- Linear ----
v_lin = v_linear_analitica(t, m, k, g)
x_lin = x_linear_analitica(t, m, k, g, x0)
t_e, v_e_lin = euler_linear(T, h, m, k, g, v0)
vt_lin = m*g/k

plt.figure()
plt.plot(t, v_lin, label='Analítica linear')
plt.plot(t_e, v_e_lin, '--', label='Euler linear')
plt.axhline(vt_lin, linestyle=':', label=f'v_t = {vt_lin:.2f} m/s')
plt.xlabel('t (s)')
plt.ylabel('v (m/s)')
plt.title('Queda com resistência linear - Bola')
plt.legend()
plt.tight_layout()
plt.show()

# ---- Quadrática ----
v_quad = v_quadratica_analitica(t, m, c, g, v0)
t_e2, v_e_quad = euler_quadratico(T, h, m, c, g, v0)
vt_quad = np.sqrt(m*g/c)

plt.figure()
plt.plot(t, v_quad, label='Analítica quadrática')
plt.plot(t_e2, v_e_quad, '--', label='Euler quadrática')
plt.axhline(vt_quad, linestyle=':', label=f'v_t = {vt_quad:.2f} m/s')
plt.xlabel('t (s)')
plt.ylabel('v (m/s)')
plt.title('Queda com resistência quadrática - Bola')
plt.legend()
plt.tight_layout()
plt.show()

