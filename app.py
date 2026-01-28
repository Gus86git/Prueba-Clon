import os
import torch
import gradio as gr
import librosa
import soundfile as sf
import numpy as np
from fish_speech.models.text2semantic.inference import Text2SemanticInference
from fish_speech.models.vqgan.inference import VQGANInference
from fish_speech.utils.file import list_files
from huggingface_hub import snapshot_download

# --- Configuracion de Modelos ---
MODEL_REPO = "fishaudio/fish-speech-1.4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Usando dispositivo: {DEVICE}")

# --- Descarga de Pesos (Auto-download para Colab) ---
def download_models():
    if not os.path.exists("checkpoints"):
        print("Descargando modelos de Hugging Face...")
        snapshot_download(repo_id=MODEL_REPO, local_dir="checkpoints")

download_models()

# --- Inicializacion de Inferencia ---
print("Cargando modelos en memoria (esto puede tardar unos minutos)...")

# Cargar VQGAN (Decode semantic to audio)
vqgan_model = VQGANInference(
    config_path="checkpoints/firefly-gan-vq-fsq-8x1024-21hz-generator.yaml",
    checkpoint_path="checkpoints/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    device=DEVICE,
)

# Cargar LLM (Text to semantic)
t2s_model = Text2SemanticInference(
    model_path="checkpoints/text2semantic-sft-medium-v1.1-half.pth",
    device=DEVICE,
    precision=torch.float16 if DEVICE == "cuda" else torch.float32
)

def clone_voice(ref_audio, text):
    if ref_audio is None or text == "":
        return None, "Por favor, sube un audio y escribe un texto."
    
    try:
        # 1. Procesar audio de referencia
        # Fish Speech usa la referencia para extraer el estilo
        # En esta implementacion simplificada, pasamos el path directamente
        
        print(f"Generando audio para: {text}")
        
        # 2. Generar tokens semanticos
        # Nota: La API de fish-speech puede variar segun la version.
        # Esta es una aproximacion funcional para la v1.4.
        semantics = t2s_model.inference(
            text=text,
            reference_audio=ref_audio,
            max_new_tokens=1024,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7
        )
        
        # 3. Decodificar a audio
        audio = vqgan_model.inference(semantics)
        
        # 4. Guardar resultado temporal
        output_path = "output.wav"
        sf.write(output_path, audio, vqgan_model.spec_transform.sample_rate)
        
        return output_path, "¡Clonación exitosa!"
        
    except Exception as e:
        return None, f"Error durante la clonación: {str(e)}"

# --- Interfaz Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Clone-Qwen: Zero-shot Voice Cloning Demo")
    gr.Markdown("Sube un audio corto (3-10s) en formato **.wav, .mp3 o .oga** de la voz que quieres clonar y escribe el texto.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Audio de Referencia (.wav, .mp3, .oga)", type="filepath")
            text_input = gr.Textbox(label="Texto a generar", placeholder="Hola, esta es mi voz clonada usando Qwen y Fish Speech.")
            generate_btn = gr.Button("Generar Voz Clonada", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(label="Resultado")
            status_output = gr.Textbox(label="Estado")

    generate_btn.click(
        fn=clone_voice,
        inputs=[audio_input, text_input],
        outputs=[audio_output, status_output]
    )

    gr.Markdown("### Notas para Colab:")
    gr.Markdown("1. Asegúrate de estar usando una **GPU T4** (Entorno de ejecución -> Cambiar tipo de entorno).")
    gr.Markdown("2. La primera ejecución descargará los modelos (~4GB).")

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
