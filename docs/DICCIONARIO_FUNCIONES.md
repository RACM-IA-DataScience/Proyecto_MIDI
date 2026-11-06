# 📖 Diccionario de Funciones y Referencia de la API

Este documento proporciona la especificación técnica exhaustiva de todas las funciones, métodos y clases implementadas en el **Proyecto MIDI**, abarcando tanto el cuaderno interactivo (`flujo_procesamiento_midi.ipynb`) como el script de ejecución modular (`flujo_procesamiento_midi_alternativa.py`).

---

## 📑 Tabla de Contenidos

### Módulo Principal: Cuaderno Interactivo (`flujo_procesamiento_midi.ipynb`)
1. [create_synthetic_demo_dataset](#1-create_synthetic_demo_dataset)
2. [scan_midi_repository](#2-scan_midi_repository)
3. [inspect_midi_file](#3-inspect_midi_file)
4. [extract_usable_training_chunks](#4-extract_usable_training_chunks)
5. [MIDILanguageModelDataset (Clase)](#5-midilanguagemodeldataset-clase)
6. [GeneradorMusicalTokens (Clase)](#6-generadormusicaltokens-clase)
7. [generar_musica_tokens](#7-generar_musica_tokens)
8. [plot_piano_roll_from_score](#8-plot_piano_roll_from_score)
9. [sintetizar_audio_midi](#9-sintetizar_audio_midi)

### Módulo Modular: Script Python (`flujo_procesamiento_midi_alternativa.py`)
10. [load_dataset](#10-load_dataset)
11. [explore_midi](#11-explore_midi)
12. [clean_data](#12-clean_data)
13. [tokenize_data](#13-tokenize_data)
14. [analyze_stats](#14-analyze_stats)
15. [MIDIDataset (Clase)](#15-mididataset-clase)
16. [prepare_data](#16-prepare_data)
17. [main](#17-main)

---

## Módulo: `flujo_procesamiento_midi.ipynb`

### 1. `create_synthetic_demo_dataset`

```python
def create_synthetic_demo_dataset(base_dir: Path) -> None
```

- **Descripción**: Genera programáticamente un corpus sintético de archivos MIDI estructurado en carpetas jerárquicas realistas. Incluye composiciones clásicas (Chopin, Bach), ensambles con percusión (Jazz, Pop) y casos borde (frecuencias fuera de rango, archivos muy cortos o corruptos) para probar la robustez del filtrado.
- **Parámetros**:
  - `base_dir` (`pathlib.Path`): Directorio raíz donde se crearán las subcarpetas y los archivos MIDI.
- **Retorno**: `None`. Los archivos se escriben directamente en el sistema de archivos.
- **Estructura generada**:
  - `classical/chopin/chopin_nocturne_op9.mid` (Piano solo, 240 notas)
  - `classical/bach/bach_two_part_invention.mid` (Invención polifónica a dos voces)
  - `jazz_quartet/autumn_leaves_jazz_trio.mid` (Multitrack con bajo y batería)
  - `pop_ballad/pop_ballad_melody.mid` (Piano y batería pop)
  - `atypical_noise/extreme_frequencies_noise.mid` (Pitches inaudibles < 21 y > 108)
  - `atypical_noise/too_short_fragment.mid` (Fragmento de 1 segundo / 8 notas)
  - `corrupted/corrupted_empty_header.mid` (Bytes inválidos para probar captura de errores)

---

### 2. `scan_midi_repository`

```python
def scan_midi_repository(root_path: Path) -> pd.DataFrame
```

- **Descripción**: Realiza una búsqueda recursiva de todos los archivos con extensiones MIDI válidas dentro del árbol de directorios especificado, construyendo un inventario tabular inicial.
- **Parámetros**:
  - `root_path` (`pathlib.Path`): Directorio base desde el cual iniciar la exploración recursiva.
- **Retorno**: `pd.DataFrame` con las siguientes columnas:
  - `filename` (`str`): Nombre del archivo con extensión.
  - `relative_path` (`str`): Ruta relativa respecto a `root_path`.
  - `file_path` (`str`): Ruta absoluta en el sistema.
  - `size_kb` (`float`): Tamaño del archivo en kilobytes.
- **Excepciones**: Manejo interno de advertencia si la carpeta raíz no existe o no contiene archivos `.mid`.

---

### 3. `inspect_midi_file`

```python
def inspect_midi_file(file_path: str) -> dict
```

- **Descripción**: Abre y analiza el contenido acústico-musical de un archivo MIDI individual utilizando el motor C++ de `symusic`.
- **Parámetros**:
  - `file_path` (`str`): Ruta del archivo MIDI a examinar.
- **Retorno**: `dict` con métricas acústicas estructuradas:
  - `filename` (`str`): Nombre del archivo.
  - `file_path` (`str`): Ruta del archivo.
  - `duration_sec` (`float`): Duración en segundos de la pieza.
  - `total_notes` (`int`): Conteo total de notas en todas las pistas.
  - `num_tracks` (`int`): Número total de pistas.
  - `is_multi_track` (`bool`): `True` si contiene más de una pista.
  - `has_drums` (`bool`): `True` si al menos una pista tiene percusión (`is_drum == True`).
  - `programs` (`list[int]`): Lista de instrumentos General MIDI presentes.
  - `pct_ideal_range` (`float`): Porcentaje de notas dentro del registro de piano [21, 108].
  - `status` (`str`): `"OK"` si la lectura fue exitosa o `"ERROR"` en caso de fallo.
  - `error_msg` (`str`): Detalle del error capturado o `None`.
- **Excepciones**: Captura genérica de excepciones `Exception` ante archivos truncados o corruptos, registrando el error sin detener el flujo global.

---

### 4. `extract_usable_training_chunks`

```python
def extract_usable_training_chunks(token_ids_list: list[list[int]], max_len: int = 256, stride: int = 128) -> list[list[int]]
```

- **Descripción**: Aplica una ventana deslizante con solapamiento sobre las listas de IDs de tokens de cada obra para transformarlas en fragmentos homogéneos de longitud fija aptos para el entrenamiento de redes neuronales.
- **Parámetros**:
  - `token_ids_list` (`list[list[int]]`): Lista donde cada elemento es la secuencia de IDs de una canción.
  - `max_len` (`int`, opcional): Longitud de la ventana de contexto. Por defecto `256`.
  - `stride` (`int`, opcional): Desplazamiento de la ventana entre fragmentos contiguos. Por defecto `128` (50% de solapamiento).
- **Retorno**: `list[list[int]]`: Colección de fragmentos discretos. Fragmentos de longitud $\ge 64$ tokens se conservan y se completan con ceros (*padding*).

---

### 5. `MIDILanguageModelDataset` (Clase)

```python
class MIDILanguageModelDataset(torch.utils.data.Dataset)
```

Especialización de `Dataset` de PyTorch para modelado de lenguaje autorregresivo del siguiente token.

#### Métodos:

- `__init__(self, sequences: list[list[int]], max_len: int = 256, pad_id: int = 0)`
  - **Parámetros**:
    - `sequences`: Lista de secuencias de tokens pre-segmentadas.
    - `max_len`: Longitud máxima del tensor.
    - `pad_id`: ID del token de relleno (padding).
  - **Lógica**: Divide cada fragmento en un par entrada-objetivo:
    - Entrada $X = [t_0, t_1, \dots, t_{N-1}]$
    - Objetivo $Y = [t_1, t_2, \dots, t_N]$ (desplazado en 1 posición)
- `__len__(self) -> int`: Retorna la cantidad de pares $X, Y$ disponibles.
- `__getitem__(self, idx: int) -> dict[str, torch.Tensor]`: Retorna un diccionario con tensores de tipo `torch.long`:
  - `input_ids`: Tensor de forma `[max_len - 1]`.
  - `target_ids`: Tensor de forma `[max_len - 1]`.

---

### 6. `GeneradorMusicalTokens` (Clase)

```python
class GeneradorMusicalTokens(torch.nn.Module)
```

Modelo de lenguaje autorregresivo basado en Redes Neuronales Recurrentes (LSTM) para predicción de secuencias simbólicas.

#### Constructor:
```python
def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2, pad_id: int = 0)
```
- **Parámetros**:
  - `vocab_size` (`int`): Cardinalidad del vocabulario de tokens (típicamente $\approx 435$).
  - `embed_dim` (`int`): Dimensión del espacio vectorial de embedding. Por defecto `128`.
  - `hidden_dim` (`int`): Cantidad de neuronas por capa recurrente LSTM. Por defecto `256`.
  - `num_layers` (`int`): Número de capas LSTM apiladas. Por defecto `2`.
  - `dropout` (`float`): Probabilidad de regularización por dropout entre capas. Por defecto `0.2`.
  - `pad_id` (`int`): Índice del token de relleno en la tabla de embeddings.

#### Método `forward`:
```python
def forward(self, x: torch.Tensor, hidden: tuple = None) -> tuple[torch.Tensor, tuple]
```
- **Parámetros**:
  - `x` (`torch.Tensor`): Tensor de entrada de forma `[Batch, Seq_Len]`.
  - `hidden` (`tuple`, opcional): Estados ocultos previos `(h_0, c_0)`.
- **Retorno**:
  - `logits` (`torch.Tensor`): Tensor de salida no normalizado de forma `[Batch, Seq_Len, Vocab_Size]`.
  - `hidden` (`tuple`): Tupla con los estados actualizados `(h_n, c_n)`.

---

### 7. `generar_musica_tokens`

```python
def generar_musica_tokens(
    modelo: torch.nn.Module,
    semilla_ids: list[int],
    pasos_a_generar: int = 300,
    temperatura: float = 0.85,
    top_k: int = 20,
    max_context: int = 255
) -> list[int]
```

- **Descripción**: Función de inferencia autorregresiva. A partir de una secuencia de tokens inicial ("semilla" o *prompt*), consulta recurrentemente el modelo para predecir y anexar nuevos tokens musicales uno a uno.
- **Parámetros**:
  - `modelo` (`torch.nn.Module`): Instancia entrenada de `GeneradorMusicalTokens`.
  - `semilla_ids` (`list[int]`): Lista de enteros representando los primeros compases.
  - `pasos_a_generar` (`int`, opcional): Número total de tokens nuevos a predecir. Por defecto `300`.
  - `temperatura` (`float`, opcional): Factor de escalado de los logits para controlar la entropía de la distribución. Por defecto `0.85`.
  - `top_k` (`int`, opcional): Restringe el muestreo estocástico a los $K$ tokens más probables. Por defecto `20`.
  - `max_context` (`int`, opcional): Longitud máxima de ventana para recortar la entrada y evitar desbordar la capacidad del modelo. Por defecto `255`.
- **Retorno**: `list[int]`: Secuencia extendida conteniendo los tokens originales de la semilla más los tokens generados.

---

### 8. `plot_piano_roll_from_score`

```python
def plot_piano_roll_from_score(score: symusic.Score, titulo: str = "Piano Roll - Composicion Generada por IA") -> None
```

- **Descripción**: Genera una representación gráfica 2D interactiva tipo Piano Roll utilizando `matplotlib.pyplot`. Representa el tiempo musical (en pulsos / beats) en el eje horizontal y el tono (pitch MIDI) en el eje vertical, resaltando el registro de 88 teclas.
- **Parámetros**:
  - `score` (`symusic.Score`): Objeto de partitura simbólica a visualizar.
  - `titulo` (`str`, opcional): Título del gráfico.
- **Retorno**: `None`. Muestra la ventana de gráfico mediante `plt.show()`.

---

### 9. `sintetizar_audio_midi`

```python
def sintetizar_audio_midi(score: symusic.Score, ruta_wav_salida: str | Path, sample_rate: int = 22050) -> Path
```

- **Descripción**: Sintetizador acústico aditivo en Python puro. Convierte las notas del objeto `symusic.Score` en audio PCM mediante osciladores armónicos y envolventes exponenciales ADSR, asegurando reproducción independiente de códecs externos o SoundFonts.
- **Parámetros**:
  - `score` (`symusic.Score`): Objeto simbólico con las pistas y notas a sintetizar.
  - `ruta_wav_salida` (`str` | `pathlib.Path`): Destino del archivo WAV generado.
  - `sample_rate` (`int`, opcional): Frecuencia de muestreo en Hz. Por defecto `22050`.
- **Retorno**: `pathlib.Path`: Ruta confirmada del archivo `.wav` exportado.
- **Algoritmo interno**:
  1. Cálculo de frecuencia fundamental: $f = 440 \times 2^{\frac{pitch - 69}{12}}$.
  2. Generación armónica: $s(t) = 0.7 \sin(2\pi f t) + 0.3 \sin(4\pi f t)$.
  3. Modulación de envolvente: $e(t) = v \cdot \exp(-3.0 \cdot t / d)$.
  4. Normalización para prevenir distorsión por recorte (*clipping*) a un factor de seguridad de $0.9$.
  5. Escritura binaria en PCM de 16 bits mediante el módulo estándar `wave`.

---

## Módulo: `flujo_procesamiento_midi_alternativa.py`

### 10. `load_dataset`

```python
def load_dataset() -> list[str]
```

- **Descripción**: Escanea recursivamente el directorio de entrada `RAW_DATASET_DIR` ('dataset_midi_raw') en busca de archivos `.mid` y `.midi`.
- **Retorno**: `list[str]`: Lista de rutas absolutas o relativas de todos los archivos encontrados.

---

### 11. `explore_midi`

```python
def explore_midi(midi_files: list[str]) -> None
```

- **Descripción**: Imprime en consola un diagnóstico acústico de los primeros 5 archivos encontrados: duración en segundos, total de notas, detección de percusión, números de programa instrumental y porcentaje en el registro de 88 teclas [21, 108].
- **Parámetros**:
  - `midi_files` (`list[str]`): Lista de rutas de archivos MIDI.

---

### 12. `clean_data`

```python
def clean_data(midi_files: list[str]) -> list[str]
```

- **Descripción**: Filtra el dataset descartando pistas de percusión, seleccionando la pista melódica de mayor densidad instrumental y aplicando los umbrales de retención: al menos 50 notas y mínimo 5 segundos de duración. Los archivos aceptados se guardan con prefijo `clean_` en `CLEAN_DATASET_DIR`.
- **Parámetros**:
  - `midi_files` (`list[str]`): Rutas de archivos a evaluar.
- **Retorno**: `list[str]`: Lista de rutas de los archivos MIDI limpios guardados.

---

### 13. `tokenize_data`

```python
def tokenize_data(cleaned_files: list[str]) -> tuple[list[miditok.TokSequence], miditok.REMI]
```

- **Descripción**: Configura e inicializa el tokenizador REMI de MidiTok y procesa cada archivo de audio limpio, convirtiéndolo en secuencias de tokens. Imprime el tamaño del vocabulario y un ejemplo del mapeo simbólico.
- **Parámetros**:
  - `cleaned_files` (`list[str]`): Lista de archivos limpios a procesar.
- **Retorno**: `tuple`:
  - `tokens_list` (`list[miditok.TokSequence]`): Lista de objetos con secuencias de tokens.
  - `tokenizer` (`miditok.REMI`): Instancia del tokenizador utilizado.

---

### 14. `analyze_stats`

```python
def analyze_stats(tokens_list: list[miditok.TokSequence]) -> list[list[int]]
```

- **Descripción**: Calcula y muestra por consola las métricas estadísticas de longitud de secuencias (mínimo, máximo, media) y extrae las secuencias con al menos 64 tokens utilizables para modelado de redes neuronales.
- **Parámetros**:
  - `tokens_list` (`list[miditok.TokSequence]`): Secuencias tokenizadas.
- **Retorno**: `list[list[int]]`: Lista de listas de IDs de tokens utilizables.

---

### 15. `MIDIDataset` (Clase)

```python
class MIDIDataset(torch.utils.data.Dataset)
```

Dataset ligero en PyTorch para secuencias autorregresivas.

#### Métodos:
- `__init__(self, sequences: list[list[int]], max_len: int = 256, pad_id: int = 0)`
  - Empareja cada secuencia en pares $(X, Y)$ con ventana máxima `max_len`, rellenando con `pad_id`.
- `__len__(self) -> int`: Retorna la longitud del conjunto de datos.
- `__getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]`: Retorna la tupla `(x, y)` con tensores `torch.long`.

---

### 16. `prepare_data`

```python
def prepare_data(usable_sequences: list[list[int]], pad_token_id: int) -> None
```

- **Descripción**: Realiza la partición de secuencias en subconjuntos de Entrenamiento (80%) y Validación (20%), construye el `MIDIDataset` y el `DataLoader` de PyTorch, y verifica las dimensiones y tipos de datos del primer lote autorregresivo.
- **Parámetros**:
  - `usable_sequences` (`list[list[int]]`): Secuencias de IDs válidas.
  - `pad_token_id` (`int`): Identificador del token de padding.

---

### 17. `main`

```python
def main() -> None
```

- **Descripción**: Punto de entrada principal para la ejecución secuencial en línea de comandos de las 6 fases de `flujo_procesamiento_midi_alternativa.py`.
