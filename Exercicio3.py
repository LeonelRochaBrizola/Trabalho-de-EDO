import numpy as np
import matplotlib.pyplot as plt

# =============================
# Parte 1 — Neurônio biológico (modelo RC simples) com Euler
# =============================

# Parâmetros do modelo RC
C = 1.0
gL = 0.3
EL = -65.0
Iinj = 10.0

# Integração
h = 1.0   # passo
T = 100.0 # tempo final
N = int(T/h)

# Estado inicial
V = np.zeros(N+1)
V[0] = -70.0

def f_RC(V):
    return -(gL/C)*(V - EL) + Iinj/C

# Integração de Euler
t = np.linspace(0, T, N+1)
for n in range(N):
    V[n+1] = V[n] + h * f_RC(V[n])

# Solução analítica para conferência
tau = C / gL
V_inf = EL + Iinj/gL
V_analit = V_inf + (V[0] - V_inf) * np.exp(-t/tau)

plt.figure()
plt.plot(t, V, label='Euler')
plt.axhline(y=V_inf, color='gray', linestyle=':', linewidth=1.5,label=f'Equilíbrio $V_\\infty = {V_inf:.2f}$ mV')
plt.plot(t, V_analit, linestyle='--', label='Analítica')
plt.xlabel('t')
plt.ylabel('V(t) [mV]')
plt.title('Modelo de membrana (RC) — Euler vs Analítica')
plt.legend()
plt.grid(True)
plt.show()

# =============================
# Parte 2 — Neurônio artificial (ReLU e Sigmoide)
# =============================

def relu(z):
    return np.maximum(0.0, z)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def neuron(x1, x2, w1, w2, b, act='relu'):
    z = w1*x1 + w2*x2 + b
    if act == 'relu':
        return relu(z)
    elif act == 'sigmoid':
        return sigmoid(z)
    else:
        return z  # linear

x1 = np.linspace(-2, 2, 200)
x2 = np.linspace(-2, 2, 200)
X1, X2 = np.meshgrid(x1, x2)

w1, w2, b = 1.0, -1.0, 0.0

Z_relu = neuron(X1, X2, w1, w2, b, act='relu')
Z_sig  = neuron(X1, X2, w1, w2, b, act='sigmoid')

plt.figure()
plt.imshow(Z_relu, extent=[x1.min(), x1.max(), x2.min(), x2.max()], origin='lower', aspect='auto')
plt.title('Saída do neurônio (ReLU)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='saída')
plt.show()

plt.figure()
plt.imshow(Z_sig, extent=[x1.min(), x1.max(), x2.min(), x2.max()], origin='lower', aspect='auto')
plt.title('Saída do neurônio (Sigmoide)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='saída')
plt.show()
