# 📊 Diagramas de Arquitectura, Flujo y Modelado UML

Este documento contiene la colección completa de especificaciones visuales y diagramas técnicos del **Proyecto MIDI**, modelados con sintaxis estándar de **Mermaid**.

---

## 📑 Índice de Diagramas
1. [Arquitectura General del Sistema](#1-arquitectura-general-del-sistema)
2. [Diagrama de Flujo del Pipeline Integral](#2-diagrama-de-flujo-del-pipeline-integral)
3. [Árbol de Decisión: Algoritmo de Limpieza y Filtrado](#3-árbol-de-decisión-algoritmo-de-limpieza-y-filtrado)
4. [Diagrama de Clases UML del Sistema](#4-diagrama-de-clases-uml-del-sistema)
5. [Diagrama de Secuencia: Inferencia Autorregresiva y Síntesis](#5-diagrama-de-secuencia-inferencia-autorregresiva-y-síntesis)
6. [Diagrama de Transición de Estados: Ciclo de Vida del Token Musical](#6-diagrama-de-transición-de-estados-ciclo-de-vida-del-token-musical)

---

## 1. Arquitectura General del Sistema

El sistema sigue una arquitectura modular en capas que desacopla la adquisición simbólica, la tokenización, el modelado neuronal de lenguaje musical y el motor de renderizado acústico.

```mermaid
flowchart TD
    subgraph S1["Capa 1: Ingesta y Datos Crudos"]
        A1["Repositorio de Archivos MIDI<br>('dataset_midi_raw/')"]
        A2["Escaneo Recursivo y Metadatos<br>('scan_midi_repository')"]
        A1 --> A2
    end

    subgraph S2["Capa 2: Procesamiento Acústico y Limpieza"]
        B1["Inspección Simbólica con symusic<br>('inspect_midi_file')"]
        B2["Filtro Polifónico y Criterios<br>(Instrumento, Duración, Rango)"]
        B3["Dataset Limpio Normalizado<br>('dataset_midi_clean/')"]
        A2 --> B1
        B1 --> B2
        B2 --> B3
    end

    subgraph S3["Capa 3: Tokenización Simbólica"]
        C1["Tokenizador MidiTok REMI<br>(pitch, velocity, duration, bar, pos)"]
        C2["Vocabulario Discreto (|V| ≈ 435 tokens)"]
        C3["Chunking y Segmentación Autorregresiva<br>(Window=256, Stride=128)"]
        B3 --> C1
        C1 --> C2
        C2 --> C3
    end

    subgraph S4["Capa 4: Modelado Neuronal"]
        D1["Dataset PyTorch<br>('MIDILanguageModelDataset')"]
        D2["Red Neuronal 'GeneradorMusicalTokens'<br>(Embedding + 2x LSTM + Linear)"]
        D3["Optimizador Adam + CrossEntropyLoss"]
        C3 --> D1
        D1 --> D2
        D2 <--> D3
    end

    subgraph S5["Capa 5: Generación y Renderizado"]
        E1["Muestreador Autorregresivo<br>(Temperatura τ + Top-K Filtering)"]
        E2["Decodificación a Score MIDI<br>('tokenizer.decode')"]
        E3["Archivo MIDI Generado<br>('results/*.mid')"]
        E4["Sintetizador Acústico Armónico ADSR"]
        E5["Audio Masterizado WAV 16-bit<br>('results/*.wav')"]
        D2 --> E1
        E1 --> E2
        E2 --> E3
        E2 --> E4
        E4 --> E5
    end
```

---

## 2. Diagrama de Flujo del Pipeline Integral

Detalla la secuencia cronológica de transformación de datos desde la lectura de archivos brutos hasta la audición de la composición generada.

```mermaid
flowchart LR
    Start(["Inicio del Pipeline"]) --> Scan["1. Escanear Archivos MIDI<br>('dataset_midi_raw')"]
    Scan --> Inspect["2. Inspección Acústica<br>(symusic: notas, tracks, pitch)"]
    Inspect --> Filter{"3. ¿Supera Filtro de Calidad?<br>(>=50 notas, >=5s, 88 teclas)"}
    
    Filter -- "No" --> Discard["Registrar Descarte con Causa<br>(df_cleaning)"]
    Filter -- "Sí" --> Keep["Guardar MIDI Monotrack Limpio<br>('dataset_midi_clean')"]
    
    Keep --> Tokenize["4. Tokenización REMI<br>(MidiTok: Bar, Pos, Pitch, Vel)"]
    Tokenize --> Stats["5. Análisis Estadístico<br>(Histogramas, Min/Max/Promedio)"]
    Stats --> Chunk["6. Segmentación en Bloques<br>(Len: 256, Stride: 128)"]
    Chunk --> DataLoader["7. DataLoader PyTorch<br>(X = t_0..N-1, Y = t_1..N)"]
    
    DataLoader --> Train["8. Entrenamiento LSTM<br>(CrossEntropyLoss, Adam)"]
    Train --> Seed["9. Selección de Semilla<br>(Prompt de 16 tokens iniciales)"]
    
    Seed --> Generate["10. Generación Autorregresiva<br>(Temperature Scaling + Top-K)"]
    Generate --> Decode["11. Reconstrucción con tokenizer.decode<br>(Genera symusic.Score)"]
    
    Decode --> ExportMIDI["12. Exportar Archivo .mid<br>('results/composicion.mid')"]
    Decode --> Synthesize["13. Síntesis Aditiva ADSR<br>(Pure Python: f0 + 2f0)"]
    
    Synthesize --> ExportWAV["14. Guardar Audio WAV 16-bit<br>('results/composicion.wav')"]
    ExportMIDI --> Finish(["Fin del Proceso"])
    ExportWAV --> Finish
```

---

## 3. Árbol de Decisión: Algoritmo de Limpieza y Filtrado

Representa la lógica condicional exhaustiva ejecutada sobre cada archivo durante la fase de depuración de datos.

```mermaid
flowchart TD
    Entry(["Archivo MIDI Candidato"]) --> C1{"¿Archivo Legible y Sintaxis Válida?"}
    
    C1 -- "No (Corrupto / EOF prematuro)" --> R1["DESCARTADO: Archivo corrupto o ilegible"]
    C1 -- "Sí" --> C2{"¿Tiene pistas melódicas?<br>(is_drum == False)"}
    
    C2 -- "No (Solo batería / percusión)" --> R2["DESCARTADO: Solo contiene pistas de percusión"]
    C2 -- "Sí" --> SelectTrack["Seleccionar pista objetivo:<br>1. Familia Piano (GM 0..7)<br>2. Pista de mayor densidad de notas"]
    
    SelectTrack --> C3{"¿Notas de pista >= 50?"}
    C3 -- "No" --> R3["DESCARTADO: Densidad insuficiente (< 50 notas)"]
    C3 -- "Sí" --> C4{"¿Duración total >= 5.0 segundos?"}
    
    C4 -- "No" --> R4["DESCARTADO: Duración excesivamente corta (< 5s)"]
    C4 -- "Sí" --> C5{"¿Porcentaje de notas fuera de rango<br>(Pitch < 21 o Pitch > 108) <= 5.0%?"}
    
    C5 -- "No (> 5% notas fuera de rango)" --> R5["DESCARTADO: Registro armónico atípico"]
    C5 -- "Sí" --> KeepClean["CONSERVADO: Extraer pista única + tempos<br>Exportar a 'dataset_midi_clean/clean_*.mid'"]
```

---

## 4. Diagrama de Clases UML del Sistema

Estructura de clases, componentes neuronales, datasets y adaptadores simbólicos.

```mermaid
classDiagram
    direction TB

    class Dataset {
        <<PyTorch Abstract>>
        +__len__() int
        +__getitem__(idx: int) Any
    }

    class nn_Module {
        <<PyTorch Module>>
        +forward(*inputs) Tensor
        +parameters() Iterator
        +to(device: Device)
    }

    class MIDILanguageModelDataset {
        -List~Tensor~ x_data
        -List~Tensor~ y_data
        -int max_len
        -int pad_id
        +__init__(sequences: List~List~int~~, max_len: int, pad_id: int)
        +__len__() int
        +__getitem__(idx: int) dict~str, Tensor~
    }

    class MIDIDataset {
        -List~Tensor~ x
        -List~Tensor~ y
        +__init__(sequences: List~List~int~~, max_len: int, pad_id: int)
        +__len__() int
        +__getitem__(idx: int) tuple~Tensor, Tensor~
    }

    class GeneradorMusicalTokens {
        -Embedding embedding
        -LSTM lstm
        -Linear fc
        -int vocab_size
        -int embed_dim
        -int hidden_dim
        -int num_layers
        -float dropout
        -int pad_id
        +__init__(vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float, pad_id: int)
        +forward(x: Tensor, hidden: tuple) tuple~Tensor, tuple~
    }

    class TokenizerConfig {
        +tuple pitch_range
        +dict beat_res
        +int num_velocities
        +bool use_chords
        +bool use_rests
        +bool use_tempos
        +bool use_time_signatures
    }

    class REMI {
        -TokenizerConfig config
        -dict vocab
        +int vocab_size
        +int pad_token_id
        +encode(score: Score) TokSequence
        +decode(tokens: List) Score
        +__call__(file_or_score: Any) TokSequence
    }

    class Score {
        <<symusic>>
        +int ticks_per_quarter
        +List~Track~ tracks
        +List~Tempo~ tempos
        +List~TimeSignature~ time_signatures
        +note_num() int
        +to(unit: str) Score
        +dump_midi(path: str)
        +from_file(path: str)$ Score
    }

    class Track {
        <<symusic>>
        +int program
        +bool is_drum
        +str name
        +List~Note~ notes
    }

    class Note {
        <<symusic>>
        +int time
        +int duration
        +int pitch
        +int velocity
    }

    Dataset <|-- MIDILanguageModelDataset
    Dataset <|-- MIDIDataset
    nn_Module <|-- GeneradorMusicalTokens
    Score "1" *-- "many" Track : tracks
    Track "1" *-- "many" Note : notes
    REMI ..> Score : encodes / decodes
    REMI *-- TokenizerConfig : configured by
    MIDILanguageModelDataset ..> GeneradorMusicalTokens : feeds batches to
```

---

## 5. Diagrama de Secuencia: Inferencia Autorregresiva y Síntesis

Detalla la interacción dinámica paso a paso entre el modelo de lenguaje, el muestreador con temperatura/Top-K, el decodificador MidiTok y el motor de síntesis de audio.

```mermaid
sequenceDiagram
    autonumber
    actor Usuario
    participant Inferencia as generar_musica_tokens()
    participant Modelo as GeneradorMusicalTokens (LSTM)
    participant Sampler as Temperatura & Top-K Sampler
    participant Decodificador as MidiTok (tokenizer.decode)
    participant Visualizador as plot_piano_roll_from_score()
    participant Sintetizador as sintetizar_audio_midi()

    Usuario->>Inferencia: Iniciar generación (semilla_ids, pasos=300, τ=0.85, TopK=20)
    activate Inferencia

    loop Durante N pasos de generación (1 hasta 300)
        Inferencia->>Modelo: forward(contexto_tokens[-max_context:])
        activate Modelo
        Modelo-->>Inferencia: logits del último paso [1, Vocab_Size]
        deactivate Modelo

        Inferencia->>Sampler: Filtrar y muestrear (logits, τ=0.85, TopK=20)
        activate Sampler
        Sampler->>Sampler: Escalar logits por temperatura (logits / 0.85)
        Sampler->>Sampler: Ordenar y truncar a los 20 más probables
        Sampler->>Sampler: Softmax y torch.multinomial(probs, 1)
        Sampler-->>Inferencia: nuevo_token_id
        deactivate Sampler

        Inferencia->>Inferencia: concatenar nuevo_token_id a secuencia
    end

    Inferencia-->>Usuario: secuencia_completa_ids (tokens generados)
    deactivate Inferencia

    Usuario->>Decodificador: decode([secuencia_completa_ids])
    activate Decodificador
    Decodificador-->>Usuario: nuevo_score (symusic.Score con notas y tiempos)
    deactivate Decodificador

    Usuario->>Visualizador: plot_piano_roll_from_score(nuevo_score)
    activate Visualizador
    Visualizador-->>Usuario: Renderizado gráfico Matplotlib (Piano Roll)
    deactivate Visualizador

    Usuario->>Sintetizador: sintetizar_audio_midi(nuevo_score, "results/*.wav", sr=22050)
    activate Sintetizador
    loop Por cada nota del Score
        Sintetizador->>Sintetizador: freq = 440 * 2^((pitch - 69) / 12)
        Sintetizador->>Sintetizador: Calcular envolvente exponencial ADSR
        Sintetizador->>Sintetizador: Sumar onda armónica (0.7*sin(2πft) + 0.3*sin(4πft))
    end
    Sintetizador->>Sintetizador: Normalizar amplitud a 0.9 (anti-clipping)
    Sintetizador->>Sintetizador: Codificar en PCM 16-bit
    Sintetizador-->>Usuario: Archivo WAV generado y ruta guardada
    deactivate Sintetizador
```

---

## 6. Diagrama de Transición de Estados: Ciclo de Vida del Token Musical

Ilustra las fases por las que transita un evento musical dentro de la representación REMI a lo largo del proceso generativo autorregresivo.

```mermaid
stateDiagram-v2
    [*] --> BarEvent : Inicio de Compás

    BarEvent --> PositionEvent : Definir posición métrica en compás (ej. Pos_0..15)
    
    PositionEvent --> PitchEvent : Emisión de nota (Pitch_21..108)
    PositionEvent --> RestEvent : Silencio musical (Rest)
    
    PitchEvent --> VelocityEvent : Definir intensidad dinámica (Velocity_1..32)
    VelocityEvent --> DurationEvent : Definir longitud temporal de la nota
    
    DurationEvent --> PitchEvent : Nota polifónica adicional en el mismo instante
    DurationEvent --> PositionEvent : Siguiente pulso rítmico dentro del compás
    DurationEvent --> BarEvent : Final de compás actual / Nuevo compás
    
    RestEvent --> PositionEvent : Fin del silencio
    RestEvent --> BarEvent : Cambio de compás tras silencio
    
    BarEvent --> [*] : Criterio de parada alcanzado (Max Tokens)
    DurationEvent --> [*] : Criterio de parada alcanzado (Max Tokens)
```
