# 🎙️ Clone-Qwen: Zero-shot Voice Cloning para Google Colab

Este repositorio contiene una aplicación de clonación de voz basada en la arquitectura **Qwen2** (vía Fish Speech 1.4). Está diseñada para ejecutarse fácilmente en **Google Colab** usando una GPU T4.

## 🚀 Requisitos Rápidos
- Una cuenta de Google para usar Colab.
- El repositorio subido a tu GitHub.

## 📂 Archivos Principales
- `app.py`: Script de la aplicación con interfaz Gradio.
- `requirements.txt`: Lista de dependencias necesarias.
- `colab_block.py`: Código listo para copiar en una celda de Colab.

## 🛠️ Cómo desplegar en Colab
1. Sube este repositorio a GitHub.
2. Abre un nuevo Notebook en [Google Colab](https://colab.research.google.com/).
3. Ve a `Entorno de ejecución` -> `Cambiar tipo de entorno` y selecciona **GPU T4**.
4. Copia el contenido de `colab_block.py` en una celda.
5. **Importante**: Cambia `REPO_URL` en el código por la URL de tu repositorio.
6. Ejecuta la celda y espera al link de Gradio (`share=True`).

## 🧠 Características Técnicas
- **Modelo**: Fish Speech 1.4 (Backbone Qwen2).
- **Inferencia**: Optimizada para `float16` en CUDA.
- **Zero-shot**: Solo necesitas entre 3 y 10 segundos de audio de referencia.
- **Formatos**: Soporte para `.wav`, `.mp3` y `.oga`.
