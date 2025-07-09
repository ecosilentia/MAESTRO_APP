# %% [markdown]
# # Masterización automática con expansión estéreo
# Este script analiza un WAV, aplica normalización, EQ, compresión y expansión estéreo
# Guarda un nuevo archivo masterizado en la misma carpeta

# %%
import librosa
import soundfile as sf
import numpy as np
from audiomentations import Compose, LoudnessNormalization, LowShelfFilter, PeakingFilter, HighShelfFilter, Limiter
import os

# %% [markdown]
# ## 1. Ruta del archivo original y de salida

# %%
input_path = "C:/Users/Daniel/Desktop/Ecosilentia/MAESTRO_APP/idea nueva1.wav"
output_path = "C:/Users/Daniel/Desktop/Ecosilentia/MAESTRO_APP/idea nueva1_master.wav"

# %% [markdown]
# ## 2. Cargar audio original

# %%
y, sr = librosa.load(input_path, sr=None, mono=False)

# Convertir a 2 canales si es mono
if y.ndim == 1:
    y = np.array([y, y])

# Separar canales
left = y[0]
right = y[1]

# Mono para análisis
y_mono = librosa.to_mono(y)

# %% [markdown]
# ## 3. Análisis automático

# %%
rms = np.sqrt(np.mean(y_mono**2))
peak = np.max(np.abs(y_mono))
dynamic_range_db = 20 * np.log10((peak + 1e-6) / (rms + 1e-6))

# Análisis espectral
S = np.abs(librosa.stft(y_mono, n_fft=2048))
freqs = librosa.fft_frequencies(sr=sr)

bass_energy = np.mean(S[freqs < 150])
mid_energy = np.mean(S[(freqs >= 150) & (freqs < 4000)])
treble_energy = np.mean(S[freqs >= 4000])
total_energy = bass_energy + mid_energy + treble_energy

bass_ratio = bass_energy / total_energy
mid_ratio = mid_energy / total_energy
treble_ratio = treble_energy / total_energy

# %% [markdown]
# ## 4. Procesamiento automático

# %%
transforms = []

# Normalización a -14 LUFS
transforms.append(LoudnessNormalization(loudness_target=-14.0))

# EQ automática
eq_gain_bass = np.clip((0.33 - bass_ratio) * 12, -4, 4)
eq_gain_mid = np.clip((0.33 - mid_ratio) * 12, -4, 4)
eq_gain_treble = np.clip((0.33 - treble_ratio) * 12, -4, 4)

transforms.append(LowShelfFilter(min_gain_db=eq_gain_bass, max_gain_db=eq_gain_bass, cut_freq=150))
transforms.append(PeakingFilter(min_gain_db=eq_gain_mid, max_gain_db=eq_gain_mid, center_freq=1000))
transforms.append(HighShelfFilter(min_gain_db=eq_gain_treble, max_gain_db=eq_gain_treble, cut_freq=4000))

# Compresión adaptativa
if dynamic_range_db > 16:
    transforms.append(Limiter(threshold_db=-1.0, ratio=10.0))
elif dynamic_range_db > 10:
    transforms.append(Limiter(threshold_db=-2.0, ratio=5.0))
else:
    transforms.append(Limiter(threshold_db=-3.0, ratio=2.0))

# %% [markdown]
# ## 5. Aplicar masterización al canal medio (mid)

# %%
pipeline = Compose(transforms)

# Mid/Side procesamiento (para expansión estéreo)
mid = (left + right) / 2
side = (left - right) / 2

# Aplicar masterización solo al canal Mid
mid_processed = pipeline(samples=mid, sample_rate=sr)

# Expandir estéreo: aumentar side (ancho)
side_amplified = side * 1.3  # Factor de ensanchamiento (ajustable)

# Reconstruir L y R
left_new = mid_processed + side_amplified
right_new = mid_processed - side_amplified

# Normalizar por si hay clipping
max_val = max(np.max(np.abs(left_new)), np.max(np.abs(right_new)))
if max_val > 1.0:
    left_new /= max_val
    right_new /= max_val

# Juntar en array estéreo
y_mastered = np.vstack([left_new, right_new])

# %% [markdown]
# ## 6. Guardar archivo WAV masterizado

# %%
sf.write(output_path, y_mastered.T, sr)
print(f"✅ Archivo guardado como:\n{output_path}")

# %%