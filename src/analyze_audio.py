import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy, ttest_ind
import warnings
warnings.filterwarnings("ignore")

def get_four_moments(signal):
    mean = np.mean(signal)
    var = np.var(signal)
    sk = skew(signal)
    kurt = kurtosis(signal)
    return mean, var, sk, kurt

def sliding_window_entropy(signal, window_size=500, step_size=50, bins=256):
    entropies = []
    # Utilizei o range dos valores globais para garantir a consistência do histograma nas janelas
    min_val, max_val = np.min(signal), np.max(signal)
    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i:i + window_size]
        hist, _ = np.histogram(window, bins=bins, range=(min_val, max_val), density=True)
        # Tive que remover os zeros para não causar erro matemático no logaritmo da entropia
        hist = hist[hist > 0]
        # Normalizando tudo para somar probabilidade 1
        hist = hist / np.sum(hist)
        ent = entropy(hist, base=2)
        entropies.append(ent)
    return np.array(entropies)

print("Carregando os áudios...")
# Forcei sr=None para garantir que o librosa mantenha o sample rate original exato dos arquivos
audio1, sr1 = librosa.load("audio01.wav", sr=None)
audio2, sr2 = librosa.load("audio02.wav", sr=None)

# Extraindo os quatro momentos estatísticos dos áudios
m1_mean, m1_var, m1_sk, m1_kurt = get_four_moments(audio1)
m2_mean, m2_var, m2_sk, m2_kurt = get_four_moments(audio2)

print("\n--- Momentos (Domínio do Tempo) ---")
print("| Áudio | Média | Variância | Assimetria | Curtose |")
print("| --- | --- | --- | --- | --- |")
def f(v): return f"{v:.6f}".replace('.', ',')
print(f"| Áudio 1 | {f(m1_mean)} | {f(m1_var)} | {f(m1_sk)} | {f(m1_kurt)} |")
print(f"| Áudio 2 | {f(m2_mean)} | {f(m2_var)} | {f(m2_sk)} | {f(m2_kurt)} |")

# Aplicando a janela deslizante de 500 amostras (salto de 50) para a Entropia de Shannon
print("\nCalculando entropia de Shannon (isso pode levar alguns instantes)...")
ent1 = sliding_window_entropy(audio1, window_size=500, step_size=50)
ent2 = sliding_window_entropy(audio2, window_size=500, step_size=50)

# Por fim, rodo o teste T de Welch para comprovar se as distribuições são iguais ou não
t_stat, p_val = ttest_ind(ent1, ent2, equal_var=False)

print("\n--- Comparação Estatística das Entropias ---")
print(f"Teste T (Welch's t-test):")
print(f"Estatística T: {t_stat:.4f}")
print(f"P-value: {p_val:.4e}")

# Plotando meus gráficos finais para o artigo
print("\nGerando gráficos...")

# Gráfico 1: Sinal no tempo
time1 = np.linspace(0, len(audio1) / sr1, num=len(audio1))
time2 = np.linspace(0, len(audio2) / sr2, num=len(audio2))

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time1, audio1, color='blue', alpha=0.8)
plt.title('Sinal no Tempo - Áudio 1')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(time2, audio2, color='red', alpha=0.8)
plt.title('Sinal no Tempo - Áudio 2')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig('src/images/audio_sinal_tempo.png')

# Gráfico 2: Entropia de Shannon
time_ent1 = np.linspace(0, len(audio1) / sr1, num=len(ent1))
time_ent2 = np.linspace(0, len(audio2) / sr2, num=len(ent2))

plt.figure(figsize=(10, 5))
plt.plot(time_ent1, ent1, label='Audio 1', alpha=0.7, color='blue')
plt.plot(time_ent2, ent2, label='Audio 2', alpha=0.7, color='red')
plt.title('Entropia de Shannon (Janela 500 amostras, Passo 50)')
plt.xlabel('Tempo (s)')
plt.ylabel('Entropia (bits)')
plt.legend()
plt.tight_layout()
plt.savefig('src/images/audio_entropia.png')

print("Análise concluída com sucesso. Imagens geradas.")
