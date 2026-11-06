import os
import symusic
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from miditok import REMI, TokenizerConfig

# Configuración de carpetas
RAW_DATASET_DIR = 'dataset_midi_raw'
CLEAN_DATASET_DIR = 'dataset_midi_clean'
os.makedirs(CLEAN_DATASET_DIR, exist_ok=True)

# ==========================================
# Parte 1: Adquisición y selección del dataset
# ==========================================
def load_dataset():
    print("--- PARTE 1: Adquisición ---")
    midi_files = []
    
    # 1 y 2. Explora la estructura y selecciona archivos MIDI
    for root, dirs, files in os.walk(RAW_DATASET_DIR):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, file))
                
    # 3. Identifica cuántos archivos fueron encontrados
    print(f"Encontrados {len(midi_files)} archivos MIDI en '{RAW_DATASET_DIR}'.\n")
    return midi_files

# ==========================================
# Parte 2: Exploración de archivos MIDI
# ==========================================
def explore_midi(midi_files):
    print("--- PARTE 2: Exploración Acústico-Musical ---")
    MIN_PITCH, MAX_PITCH = 21, 108 # Rango ideal de 88 teclas (A0 a C8)
    
    for file in midi_files[:5]: # Mostramos solo los primeros 5 para no saturar la consola
        try:
            score = symusic.Score.from_file(file)
            # Analiza duración y cantidad de notas
            duration_sec = float(score.to("second").end())
            total_notes = score.note_num()
            
            # Identifica instrumentos y percusión
            has_drums = any(t.is_drum for t in score.tracks)
            programs = list(set(t.program for t in score.tracks if not t.is_drum))
            is_multi = len(score.tracks) > 1
            
            # Examina características relevantes (rango de notas)
            all_pitches = [n.pitch for t in score.tracks for n in t.notes]
            if all_pitches:
                in_range = sum(MIN_PITCH <= p <= MAX_PITCH for p in all_pitches)
                pct_in_range = (in_range / len(all_pitches)) * 100
            else:
                pct_in_range = 0.0

            print(f"[{os.path.basename(file)}]")
            print(f"  - Duración: {duration_sec:.1f}s | Notas: {total_notes}")
            print(f"  - Multi-pista: {is_multi} | Percusión: {has_drums} | Instrumentos (GM): {programs}")
            print(f"  - Notas en rango ideal (21-108): {pct_in_range:.1f}%\n")
        except Exception as e:
            print(f"[{os.path.basename(file)}] Error de lectura: {e}\n")

# ==========================================
# Parte 3: Limpieza y preparación
# ==========================================
def clean_data(midi_files):
    print("--- PARTE 3: Limpieza y Filtrado ---")
    cleaned_files = []
    descartados = 0
    
    for file in midi_files:
        try:
            score = symusic.Score.from_file(file)
            # Criterio: Rechazar pistas de percusión y priorizar la pista con más notas
            candidate_tracks = [t for t in score.tracks if not t.is_drum]
            if not candidate_tracks:
                descartados += 1
                continue
                
            # Selecciona una pista instrumental por canción (la de mayor densidad)
            selected_track = max(candidate_tracks, key=lambda t: len(t.notes))
            
            # Descarta si no cumple requisitos (muy corta o pocas notas)
            if len(selected_track.notes) < 50 or float(score.to("second").end()) < 5.0:
                descartados += 1
                continue
                
            # Guardar archivo limpio
            clean_score = symusic.Score(score.ticks_per_quarter)
            clean_score.tracks.append(selected_track)
            clean_score.tempos = score.tempos
            clean_score.time_signatures = score.time_signatures
            
            clean_path = os.path.join(CLEAN_DATASET_DIR, f"clean_{os.path.basename(file)}")
            clean_score.dump_midi(clean_path)
            cleaned_files.append(clean_path)
            
        except Exception:
            descartados += 1 # Archivos que no puedan procesarse (corruptos)

    print(f"Archivos conservados (limpios): {len(cleaned_files)}")
    print(f"Archivos descartados: {descartados}\n")
    return cleaned_files

# ==========================================
# Parte 4: Tokenización
# ==========================================
def tokenize_data(cleaned_files):
    print("--- PARTE 4: Tokenización REMI ---")
    config = TokenizerConfig(pitch_range=(21, 109), beat_res={(0, 4): 8, (4, 12): 4})
    tokenizer = REMI(config)
    
    tokens_list = []
    for file in cleaned_files:
        seq = tokenizer(file)
        if isinstance(seq, list): seq = seq[0]
        tokens_list.append(seq)
        
    vocab_size = tokenizer.vocab_size
    print(f"Tamaño del vocabulario: {vocab_size} tokens únicos.")
    
    if tokens_list and len(tokens_list[0].tokens) > 0:
        print("Ejemplo de mapeo (primeros eventos de la primera canción):")
        # Limito la iteración a la cantidad real de tokens para no exceder el índice de la lista
        limite = min(5, len(tokens_list[0].tokens))
        for i in range(limite):
            print(f"  ID: {tokens_list[0].ids[i]} -> Token: {tokens_list[0].tokens[i]}")
    print()
    return tokens_list, tokenizer

# ==========================================
# Parte 5: Análisis estadístico
# ==========================================
def analyze_stats(tokens_list):
    print("--- PARTE 5: Análisis Estadístico ---")
    if not tokens_list:
        return []
        
    lengths = [len(seq.ids) for seq in tokens_list]
    print(f"Longitud Mínima: {np.min(lengths)}")
    print(f"Longitud Máxima: {np.max(lengths)}")
    print(f"Longitud Promedio: {np.mean(lengths):.2f}")
    
    # Identifica secuencias utilizables (ej. mayor a 64 tokens)
    usable_sequences = [seq.ids for seq in tokens_list if len(seq.ids) >= 64]
    print(f"Secuencias utilizables para entrenamiento (>= 64 tokens): {len(usable_sequences)}\n")
    
    # Genera un histograma de distribución
    try:
        plt.figure(figsize=(8, 4))
        plt.hist(lengths, bins=15, color='#3498db', edgecolor='black')
        plt.title('Distribución de Longitudes de Secuencia (Tokens)')
        plt.xlabel('Número de Tokens')
        plt.ylabel('Cantidad de Archivos')
        plt.grid(axis='y', alpha=0.75)
        print("Cerrar la ventana del gráfico para continuar con la ejecución...")
        plt.show()
    except Exception as e:
        print("No se pudo mostrar el gráfico en este entorno.")
        
    return usable_sequences

# ==========================================
# Parte 6: Preparación para Entrenamiento
# ==========================================
class MIDIDataset(Dataset):
    def __init__(self, sequences, max_len=256, pad_id=0):
        self.x, self.y = [], []
        for seq in sequences:
            cur = seq[:max_len]
            x_tok, y_tok = cur[:-1], cur[1:] # X = input, Y = target desplazado
            pad_len = (max_len - 1) - len(x_tok)
            self.x.append(torch.tensor(x_tok + [pad_id] * pad_len, dtype=torch.long))
            self.y.append(torch.tensor(y_tok + [pad_id] * pad_len, dtype=torch.long))
            
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def prepare_data(usable_sequences, pad_token_id):
    print("--- PARTE 6: Preparación PyTorch ---")
    if not usable_sequences:
        print("No hay datos suficientes para preparar.")
        return
        
    # Divide las canciones
    random.seed(42)
    random.shuffle(usable_sequences)
    split_idx = int(0.8 * len(usable_sequences))
    
    train_data = usable_sequences[:split_idx]
    val_data = usable_sequences[split_idx:]
    
    print(f"Canciones en entrenamiento: {len(train_data)}")
    print(f"Canciones en validación: {len(val_data)}")
    
    # Verifica que los datos estén correctamente preparados (Dimensiones de los tensores)
    train_dataset = MIDIDataset(train_data, max_len=256, pad_id=pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    if len(train_dataset) > 0:
        sample_x, sample_y = next(iter(train_loader))
        print(f"\nVerificación del DataLoader (Lote Autoregresivo):")
        print(f"  Dimensiones de entrada (X): {sample_x.shape} -> [Batch, Seq_Len]")
        print(f"  Dimensiones objetivo (Y): {sample_y.shape} -> [Batch, Seq_Len]")
        print(f"  Tipo de dato del tensor: {sample_x.dtype}")
        print("¡Los datos están correctamente dimensionados y listos para una Red Neuronal!")

# ==========================================
# Ejecución Principal
# ==========================================
def main():
    midi_files = load_dataset()
    if not midi_files:
        return
    explore_midi(midi_files)
    cleaned_files = clean_data(midi_files)
    tokens_list, tokenizer = tokenize_data(cleaned_files)
    usable_sequences = analyze_stats(tokens_list)
    
    # Pasamos el pad_id del tokenizador para rellenar los tensores
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
    prepare_data(usable_sequences, pad_token_id=pad_id)

if __name__ == "__main__":
    main()