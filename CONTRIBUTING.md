# 🤝 Guía de Contribución al Proyecto MIDI

¡Gracias por tu interés en colaborar con este proyecto de música simbólica y modelado neuronal! Para mantener la calidad del código y la coherencia metodológica, te invitamos a seguir estas pautas.

---

## 🛠 Entorno de Desarrollo y Configuración

1. Haz un fork del repositorio y clona tu copia localmente:
   ```bash
   git clone https://github.com/tu-usuario/Proyecto_MIDI.git
   cd Proyecto_MIDI
   ```
2. Crea un entorno virtual y asegúrate de instalar las dependencias exactas:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # O .\.venv\Scripts\Activate.ps1 en Windows
   pip install -r requirements.txt
   ```

---

## 📋 Reglas de Desarrollo y Buenas Prácticas

- **Preservación del Rango de 88 Teclas**: Cualquier nuevo filtro o transformador debe respetar el estándar de piano acústico (alturas MIDI entre 21 y 108).
- **Compatibilidad de Tokenización**: Si modificas la configuración de `MidiTok`, documenta si el cambio expande o contrae la cardinalidad del vocabulario (`tokenizer.vocab_size`).
- **Reproducibilidad en Síntesis**: El motor de audio en `sintetizar_audio_midi` debe mantenerse en Python puro / NumPy para evitar introducir dependencias de sistema difíciles de compilar en entornos portables.
- **Exclusión de Archivos Temporales y Privados**: Nunca subas archivos en la carpeta `ignorar/`, entornos virtuales o archivos pesados no procesados.

---

## 🚀 Flujo para Pull Requests

1. Crea una rama descriptiva para tu funcionalidad:
   ```bash
   git checkout -b feature/nueva-funcionalidad-o-mejora
   ```
2. Realiza tus commits siguiendo el estándar de [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat: agregar nuevo extractor de acordes`
   - `fix: corregir cálculo de envolvente ADSR`
   - `docs: actualizar diagrama de secuencia en DIAGRAMAS.md`
3. Asegúrate de que el cuaderno y los scripts ejecuten sin errores.
4. Abre un Pull Request describiendo detalladamente la motivación del cambio y pruebas realizadas.
