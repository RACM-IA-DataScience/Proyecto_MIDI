# 📘 Documento Técnico: Arquitectura y Modelado de Música Simbólica con MidiTok (REMI) y PyTorch

---

## 1. Introducción y Fundamentos de la Música Simbólica

La generación musical computacional mediante Inteligencia Artificial se divide fundamentalmente en dos paradigmas:
1. **Audio en Crudo (*Raw Audio*)**: Procesamiento de ondas muestreadas a altas frecuencias (por ejemplo, $44.1\text{ kHz}$ o $48\text{ kHz}$), donde generar un segundo de música requiere modelar decenas de miles de muestras continuas.
2. **Música Simbólica (*Symbolic Music*)**: Representación estructural de los eventos acústicos mediante notación discreta de alto nivel (frecuencia fundamental, tiempo de inicio, duración, dinámica y timbre), formalizada universalmente a través del protocolo **MIDI (Musical Instrument Digital Interface)**.

El presente proyecto implementa un flujo de trabajo integral (*end-to-end*) optimizado para **Música Simbólica**. Al abstraer los detalles micro-acústicos de la forma de onda, el sistema se enfoca en capturar las dependencias temporales de la estructura musical (melodía, armonía, métrica, polifonía y ritmo).

---

## 2. Representaciones Simbólicas: Piano Roll vs. Tokenización REMI

### 2.1 Limitaciones del Paradigma Clásico Matricial (Piano Roll)
Históricamente, los sistemas de aprendizaje profundo para música utilizaban matrices bidimensionales tipo **Piano Roll** $M \in \mathbb{R}^{T \times P}$, donde $T$ representa los pasos temporales discretizados y $P$ las 128 notas MIDI posibles:
- **Alta Dispersión (*Sparsity*)**: Más del $95\%$ de las entradas de una matriz de piano roll corresponden a ceros (silencios), lo que produce un desperdicio severo de memoria y capacidad de cómputo.
- **Pérdida de Dinámica e Información Expresiva**: La velocidad (*velocity*) suele binizarse o truncarse a valores constantes, perdiendo la expresividad interpretativa del intérprete humano.
- **Dificultad de Reconstrucción**: Transformar una matriz continua o binaria de vuelta a un archivo `.mid` válido suele provocar notas flotantes, solapamientos incoherentes y artefactos rítmicos.

### 2.2 El Paradigma Moderno: Tokenización Basada en Eventos (REMI)
Para superar las deficiencias del Piano Roll, este proyecto adopta el protocolo de tokenización **REMI (*Revamped MIDI representation*)** implementado a través de la librería especializada `MidiTok`. REMI modela la música de manera equivalente a un texto en Procesamiento del Lenguaje Natural (NLP):

| Tipo de Token | Descripción Musical | Rango / Valores Típicos |
|---|---|---|
| `Bar_None` | Delimitador explícito de inicio de compás | 1 token global |
| `Position_X` | Subdivisión rítmica dentro del compás (cuantización métrica) | $X \in [0, 15]$ (resolución de 16vos) |
| `Pitch_X` | Altura tonal de la nota según la escala cromática estándar | $X \in [21, 108]$ (registro de piano de 88 teclas) |
| `Velocity_X` | Fuerza de ataque / dinámica de la nota cuantizada en bins | 32 niveles uniformes |
| `Duration_X` | Tiempo que la nota permanece pulsada antes del apagado (*Note-Off*) | Subdivisions cuantizadas en ticks |
| `Rest_X` | Intervalo de silencio explícito entre compases o eventos | Valores métricos de pausa |

#### Ventajas Cruciales de REMI:
1. **Espacio de Estados Compacto**: Reduce el vocabulario a aproximadamente 435 tokens únicos ($|V| \approx 435$), permitiendo el uso de arquitecturas de modelado de lenguaje autorregresivo eficientes.
2. **Reversibilidad Exacta (*Lossless Symbolic Round-Trip*)**: La decodificación mediante `tokenizer.decode()` reconstruye la partitura (`symusic.Score`) preservando armaduras, compases y duraciones exactas.

---

## 3. Pipeline de Adquisición y Auditoría de Datos

El módulo de ingesta recorre recursivamente el directorio raíz (`dataset_midi_raw/`) reconociendo extensiones `.mid`, `.midi`, `.MID` y `.MIDI`. Cada archivo es sometido a una auditoría preliminar mediante `symusic`, extrayendo los siguientes vectores de características:

1. **Duración Temporal ($D_{sec}$)**:
   $$D_{sec} = \text{score.to}("second").end()$$
2. **Conteo Total de Notas ($N_{total}$)**:
   $$N_{total} = \sum_{k=1}^{K} |\{n \in \text{Track}_k\}|$$
3. **Distribución Polifónica y Pistas**: Detección de pistas melódicas e instrumentales vs pistas de percusión (`is_drum == True`).
4. **Porcentaje en Registro Estándar de Piano ($P_{range}$)**:
   Calculado sobre el conjunto de alturas de nota $\{p_i\}$:
   $$P_{range} = \frac{|\{p_i \mid 21 \le p_i \le 108\}|}{|\{p_i\}|} \times 100$$
   donde $21 = \text{A}_0$ ($27.5\text{ Hz}$) y $108 = \text{C}_8$ ($4186.0\text{ Hz}$).

---

## 4. Algoritmo de Limpieza, Filtrado y Normalización

Para garantizar que el modelo entrene con datos de alta coherencia armónica y formal, se aplica un pipeline de descarte riguroso con los siguientes criterios matemáticos y de teoría musical:

```
                  [Archivo MIDI]
                        │
       ¿Es legible por symusic y sin fallos?
            ├── No  ──> [DESCARTADO: Corrupto / Ilegible]
            └── Sí
                 │
       ¿Contiene pistas instrumentales (no percusión)?
            ├── No  ──> [DESCARTADO: Solo batería]
            └── Sí
                 │
     Seleccionar pista de mayor densidad (GM 0-7 priorizado)
                 │
       ¿Cantidad de notas >= 50?
            ├── No  ──> [DESCARTADO: Poca densidad melódica]
            └── Sí
                 │
       ¿Duración >= 5.0 segundos?
            ├── No  ──> [DESCARTADO: Fragmento excesivamente corto]
            └── Sí
                 │
       ¿Notas fuera del rango [21, 108] <= 5.0%?
            ├── No  ──> [DESCARTADO: Ruido atípico fuera de 88 teclas]
            └── Sí
                 │
       [CONSERVADO: Exportar pista única a 'dataset_midi_clean/']
```

### Justificación de los Criterios:
- **Exclusión de Pistas de Percusión (`is_drum`)**: Las pistas de ritmo no poseen tono cromático; sus números de nota mapean instrumentos físicos (bombo, tarola, hi-hat) que corromperían las relaciones armónicas del modelo tonal.
- **Umbral de 50 Notas y 5 Segundos**: Filtra efectos de sonido incidentales o archivos de prueba vacíos que no aportan estructuras de fraseo musical.
- **Tolerancia del 5% Fuera de Rango**: Protege el vocabulario contra archivos de sintetizadores microtonales o transposiciones erróneas en subgraves inaudibles.

---

## 5. Tokenización con MidiTok REMI

El tokenizador se instancia con la siguiente configuración técnica:

```python
tokenizer_config = TokenizerConfig(
    pitch_range=(21, 109),                  # 88 teclas estándar
    beat_res={(0, 4): 8, (4, 12): 4},       # Cuantización rítmica adaptativa
    num_velocities=32,                      # 32 niveles de dinámica
    use_chords=True,                        # Detección de acordes simultáneos
    use_rests=True,                         # Inclusión de silencios métricos
    use_tempos=True,                        # Cambios de tempo
    use_time_signatures=True,               # Compases métricos
    use_programs=False,                     # Monotímbrico (Piano acústico)
)
```

Cada archivo limpio es procesado produciendo una secuencia unidimensional de identificadores enteros:
$$\mathbf{s} = [t_1, t_2, t_3, \dots, t_L], \quad t_i \in \{0, 1, \dots, |V|-1\}$$

---

## 6. Preparación de Secuencias y Modelado de Lenguaje

### 6.1 Ventana Deslizante con Solapamiento (*Chunking*)
Las secuencias musicales exhiben longitudes muy variables. Para adecuar los datos a tensores homogéneos de longitud fija $L_{max} = 256$, se aplica una segmentación con solapamiento (*stride* $S = 128$, es decir, $50\%$ de overlap):
$$\mathbf{c}_k = \mathbf{s}[k \cdot S : k \cdot S + L_{max}]$$
Si el fragmento residual posee una longitud $\ge 64$ tokens pero $< L_{max}$, se aplica relleno (*padding*) con un identificador especial $t_{pad} = 0$.

### 6.2 Pareamiento Autorregresivo de Entrada y Objetivo
Para formular el problema como modelado de lenguaje autorregresivo del siguiente token (*Next-Token Prediction*):
$$\mathbf{X} = [t_0, t_1, \dots, t_{N-1}]$$
$$\mathbf{Y} = [t_1, t_2, \dots, t_N]$$
donde $\mathbf{Y}$ representa la secuencia desplazada exactamente una posición temporal hacia el futuro.

### 6.3 Prevención de Contaminación (*Song-Level Split*)
La partición en conjuntos de **Entrenamiento (80%)** y **Validación (20%)** se ejecuta a nivel de archivo/canción antes del chunking. Esto previene que fragmentos de una misma obra aparezcan simultáneamente en entrenamiento y evaluación (*data leakage*).

---

## 7. Arquitectura de la Red Neuronal Generativa (`GeneradorMusicalTokens`)

El modelo `GeneradorMusicalTokens` está diseñado específicamente para procesar secuencias discretas de tokens musicales:

```
Entrada: X [Batch, Seq_Len]
   │
   ▼
┌────────────────────────────────────────────────────────┐
│ Capa de Embedding (nn.Embedding)                       │
│ Dimensión de Entrada: |V| ≈ 435                         │
│ Dimensión de Proyección: 128                            │
│ padding_idx = pad_id                                   │
└────────────────────────────────────────────────────────┘
   │  Tensor [Batch, Seq_Len, 128]
   ▼
┌────────────────────────────────────────────────────────┐
│ Capa Recurrente LSTM Multicapa (nn.LSTM)               │
│ Capas: 2 | Tamaño Oculto: 256                          │
│ Dropout Inter-capa: 0.2 | batch_first = True           │
└────────────────────────────────────────────────────────┘
   │  Tensor [Batch, Seq_Len, 256]
   ▼
┌────────────────────────────────────────────────────────┐
│ Capa Lineal Clasificadora (nn.Linear)                  │
│ Entrada: 256 ──> Salida: |V| Logits                    │
└────────────────────────────────────────────────────────┘
   │
Salida: Logits [Batch, Seq_Len, |V|]
```

### 7.1 Formulación Matemática del Entrenamiento
La red se entrena optimizando los pesos $\theta$ mediante el algoritmo **Adam** con tasa de aprendizaje $\eta = 0.003$ y recorte de gradientes ($\|\mathbf{g}\|_2 \le 1.0$). La función de pérdida es la **Entropía Cruzada Categórica**:

$$\mathcal{L}(\theta) = -\frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} \log P(y_i \mid \mathbf{x}_{\le i}; \theta)$$

donde $\mathcal{T} = \{i \mid y_i \neq t_{pad}\}$ omite estrictamente los tokens de relleno mediante el argumento `ignore_index=pad_id`.

---

## 8. Estrategias de Inferencia: Muestreo con Temperatura y Top-K

Para generar música expresiva evitando repeticiones mecánicas o notas atípicas incoherentes, el proceso de decodificación autorregresiva implementa dos filtros estocásticos:

### 8.1 Escalado por Temperatura ($\tau$)
Dada la salida de logits $\mathbf{z} \in \mathbb{R}^{|V|}$ generada por el modelo para el siguiente paso, se aplica una temperatura $\tau > 0$:

$$z'_i = \frac{z_i}{\tau}$$

- $\tau \to 0$: Aproxima la decodificación voraz (*greedy decoding*), generando melodías monótonas y altamente repetitivas.
- $\tau > 1$: Distribución casi uniforme que introduce notas caóticas.
- $\tau \in [0.80, 0.85]$: Equilibrio óptimo entre coherencia estructural y creatividad melódica.

### 8.2 Filtrado Top-$K$ ($K = 20$)
Se seleccionan únicamente los $K$ tokens con mayor valor de $z'_i$, asignando $-\infty$ a los demás antes de calcular la distribución de probabilidad con la función Softmax:

$$P(w_i) = \frac{\exp(z'_i)}{\sum_{j \in \text{Top-K}} \exp(z'_j)}$$

El siguiente token se muestrea estocásticamente a partir de esta distribución mediante `torch.multinomial(probs, 1)`.

---

## 9. Síntesis de Audio Acústica en Python Puro

A diferencia de las arquitecturas que dependen de binarios externos compilados (`FluidSynth`, dependencias C++ complejas o SoundFonts pesados que a menudo fallan en entornos portables), este proyecto implementa un sintetizador aditivo directo en Python puro y NumPy (`sintetizar_audio_midi`):

### 9.1 Frecuencia Fundamental de Cada Nota
La altura MIDI $p \in [21, 108]$ se transforma a frecuencia fundamental $f_0$ en Hertz según la afinación temperada estándar ($A_4 = 440\text{ Hz}$, nota 69):
$$f_0 = 440.0 \times 2^{\frac{p - 69}{12}}$$

### 9.2 Generación Armónica y Envolvente Acústica
Para cada nota de duración $d$ y velocidad normalizada $v = \frac{\text{velocity}}{127}$:
1. **Onda Armónica Multitono**:
   $$s(t) = 0.7 \sin(2\pi f_0 t) + 0.3 \sin(2\pi (2f_0) t)$$
   La presencia del segundo armónico ($2f_0$) aporta cuerpo tímbrico, emulando la resonancia de las cuerdas de un piano acústico.
2. **Envolvente Exponencial ADSR**:
   $$E(t) = v \cdot \exp\left(-3.0 \cdot \frac{t}{d}\right)$$
   Garantiza un ataque inmediato y un decaimiento gradual natural sin chasquidos (*clicks*) en los bordes de la onda.
3. **Normalización Anti-Clipping y Guardado WAV**:
   $$\mathbf{y}_{norm} = 0.9 \cdot \frac{\mathbf{y}}{\max(|\mathbf{y}| + \epsilon)}$$
   La señal continua se cuantiza en enteros con signo de 16 bits (rango $[-32767, 32767]$) y se exporta como archivo WAV estándar con frecuencia de muestreo de $22050\text{ Hz}$.

---

## 10. Métricas, Visualización y Resultados

El proyecto incluye dos modalidades de inspección analítica:
1. **Visualizador de Piano Roll**: Renderizado matricial bidimensional con Matplotlib mediante rectángulos proporcionales a la duración y altura de cada nota, señalando visualmente el rango de 88 teclas.
2. **Reproducción Interactiva de Audio**: Integración directa del reproductor HTML5 `IPython.display.Audio` para escuchar el archivo `.wav` generado sin salir del cuaderno de trabajo.

---

## 11. Conclusiones y Trabajo Futuro

El pipeline demuestra que la representación por eventos REMI supera holgadamente el modelado matricial tradicional de piano roll en términos de velocidad, compacidad y calidad musical. 

### Posibles Extensiones:
- **Modelos Transformer Decoders**: Migración de la arquitectura LSTM hacia decodificadores autorregresivos basados en autoatención (como GPT-2 o LLaMA de tamaño reducido).
- **Entrenamiento Condicionado por Estilo**: Incorporación de tokens de prefijo (*prompt tokens*) que condicionen el género musical, el tempo deseado o la tonalidad.
- **Datasets de Escala Masiva**: Integración de colecciones internacionales de piano solista (por ejemplo, el corpus **MAESTRO** de Google Magenta).
