from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import os
from dotenv import load_dotenv
import sys


from src.movenet import predict_movenet_for_image
from src.pguardian import predict

load_dotenv()
# Obtener el token de acceso a la API de Telegram desde la variable de entorno
TOKEN = os.getenv("BOT_TOKEN")

# Definir una función para el comando /start
def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Hola pancho! Soy un bot que te ayudará a detectar si tenés los brazos arriba o abajo en una foto. \U0001F32D\n\nEnvíame una foto y te diré si tenés los brazos arriba o abajo.")

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
    keypoints = predict_movenet_for_image(image_path, "./bot/images/output.jpg" ,process_image=True)

    label,score = predict(keypoints)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open("./bot/images/output.jpg", 'rb'))
    
    # Enviar un mensaje de confirmación al usuario
    update.message.reply_text("Tus brazos están " + label + " con un " + str(score) + "% de confianza. \U0001F32D")






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
dispatcher.add_handler(image_handler)

# Registrar el controlador para comandos desconocidos
unknown_handler = MessageHandler(Filters.command, unknown)
dispatcher.add_handler(unknown_handler)

# Iniciar el bot
updater.start_polling()

# Mantener el bot en ejecución
updater.idle()
