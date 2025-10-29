import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
V = 1.0          # volume do reator (L)
r = 0.05         # vazão volumétrica (L/min)
Cin = 2.0        # concentração de entrada (g/L)
C0 = 0.0         # concentração inicial (g/L)
h = 0.1          # passo de integração (min)
tempF = 100.0    # tempo final (min)

# Vetor de tempo
t = np.arange(0.0, tempF + h, h)

# Constante do sistema
k = r / V

# Solução analítica
C_analitico = Cin - (Cin - C0) * np.exp(-k * t)

# Solução numérica (método de Euler)
C_euler = np.zeros_like(t)
C_euler[0] = C0
for n in range(len(t) - 1):
    dCdt = (r / V) * (Cin - C_euler[n])
    C_euler[n + 1] = C_euler[n] + h * dCdt

# =============================================================
# Gráfico 1 – Concentração C(t) (Analítica × Euler)
# =============================================================
plt.figure(figsize=(8, 5))
plt.plot(t, C_analitico, label='Solução Analítica', linewidth=2)
plt.plot(t, C_euler, '--', label='Método de Euler (h=0.1)', linewidth=1.8)
plt.xlabel('Tempo (min)')
plt.ylabel('Concentração C(t) [g/L]')
plt.title('Mistura em tanque Evolução da Concentração no Tempo')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()


# =============================================================
# Gráfico 2 – Erro absoluto entre as soluções
# =============================================================
erro_abs = np.abs(C_analitico - C_euler)
plt.figure(figsize=(8, 5))
plt.plot(t, erro_abs, color='red', label='Erro |Analítico - Euler|')
plt.xlabel('Tempo (min)')
plt.ylabel('Erro absoluto [g/L]')
plt.title('Erro absoluto ao longo do tempo (Euler h=0.1)')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()
