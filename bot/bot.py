from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import os
from dotenv import load_dotenv
import sys

import numpy as np

from app import app

from src.movenet import predict_movenet_for_image
from src.pguardian import predict

load_dotenv()
# Obtener el token de acceso a la API de Telegram desde la variable de entorno
TOKEN = os.getenv("BOT_TOKEN")

# Definir una función para el comando /start
def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Hola pancho! Soy un bot que te ayudará a detectar si realizas correctamente ejercicios de peso muerto. \U0001F32D\n\nEnvíame una foto  o video y te diré si tu postura es correcta o no. \U0001F4F7")

# Definir una función para manejar mensajes desconocidos
def unknown(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Lo siento, no entiendo ese comando.")# Función para manejar los mensajes que contienen imágenes

def handle_image(update, context):
    # Obtener la instancia del archivo de imagen
    photo = update.message.photo[-1].get_file()
    
    # Generar un nombre único para el archivo de imagen
    file_name = os.path.join('./bot/images', f'{photo.file_id}.jpg')
    
    # Descargar la imagen en el directorio 'images'
    photo.download(file_name)

    # Obtener la ruta del archivo de imagen
    image_path = os.path.join(os.getcwd(), file_name)

    # Obtener los keypoints de la imagen
    keypoints, overlay = predict_movenet_for_image(image_path, "./bot/images/output.jpg" ,process_image=True)

    label,score = predict(keypoints)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open("./bot/images/output.jpg", 'rb'))
    
    # Enviar un mensaje de confirmación al usuario
    update.message.reply_text("Tu postura es " + label + " con un " + str(score) + "% de confianza. \U0001F32D")


def handle_video(update, context):

    
    # Obtener el objeto de mensaje de video del mensaje recibido
    video_message = update.message.video
    
    # Obtener información del video
    duration = video_message.duration
    width = video_message.width
    height = video_message.height
    message = "Tu video tiene una duración de " + str(duration) + " segundos, con un ancho de " + str(width) + " y un alto de " + str(height) + " pixeles. \U0001F4F7"

    # Responder al usuario con un mensaje de confirmación
    update.message.reply_text(message)
    update.message.reply_text("Analizando video... \U0001F50D")

    # Desacargar el video en el directorio 'videos'
    video_file_name = os.path.join('./bot/videos', f'{video_message.file_id}.mp4')
    video_message.get_file().download(video_file_name)

    print(video_file_name)
    # Crear un objeto de VideoCapture desde el array de video
    predictions, score = app.analize_video(video_file_name)

    # Que los porcentajes de malas posturas sean de 2 decimales
    response_message = "Tu postura es " + str(round(predictions,2))  + "% con un " + str(round(score,2)) + "% de confianza. \U0001F32D"
    # Responder al usuario con un mensaje de confirmación
    update.message.reply_text(response_message)

# Crear el objeto Updater y pasarle el token
updater = Updater(token=TOKEN, use_context=True)

# Obtener el despachador para registrar los controladores de eventos
dispatcher = updater.dispatcher

# Registrar el controlador para el comando /start
start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

# Registrar el controlador para mensajes de texto
# message_handler = MessageHandler(Filters.text & (~Filters.command), echo)
# dispatcher.add_handler(message_handler)

# Registrar el controlador para mensajes que contienen imágenes
image_handler = MessageHandler(Filters.photo, handle_image)
dispatcher.add_handler(MessageHandler(Filters.video, handle_video))
dispatcher.add_handler(image_handler)

# Registrar el controlador para comandos desconocidos
unknown_handler = MessageHandler(Filters.command, unknown)
dispatcher.add_handler(unknown_handler)

# Iniciar el bot
updater.start_polling()

# Mantener el bot en ejecución
updater.idle()
