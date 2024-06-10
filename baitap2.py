import json
import nltk
from nltk.stem import WordNetLemmatizer
import random
import telepot
from telepot.loop import MessageLoop
import tensorflow as tf
import numpy as np
import cv2
import os
import requests

# Load intents file
with open('intents.json') as file:
    data = json.load(file)
# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load TFLite model and allocate tensors
interpreter1 = tf.lite.Interpreter(model_path="text.tflite")
interpreter1.allocate_tensors()

# Get input and output tensors details
input_details = interpreter1.get_input_details()
output_details = interpreter1.get_output_details()

# Extract words and classes from intents
words = [lemmatizer.lemmatize(w.lower()) for w in sorted(list(set([word.lower() for intent in data['intents'] for pattern in intent['patterns'] for word in nltk.word_tokenize(pattern)]))) if w not in ['?', '!', '.', ',']]
classes = sorted(list(set([intent['tag'] for intent in data['intents']])))

# Define helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag, dtype=np.float32).reshape(1, -1)

def predict_class(sentence):
    p = bow(sentence, words)
    interpreter1.set_tensor(input_details[0]['index'], p)
    interpreter1.invoke()
    res = interpreter1.get_tensor(output_details[0]['index'])[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(ints, intents_json):
    if not ints:  # Nếu danh sách ints rỗng
        return "Sorry, I don't understand"
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            response = random.choice(i['responses'])
            if '{temp}' in response:
                temp = 25  # Giả định nhiệt độ là 25 độ Celsius
                response = response.replace('{temp}', str(temp))
            return response
    return "Sorry, I don't understand"
def chatbot_response(user_input):
    ints = predict_class(user_input)
    return get_response(ints, data)



#Nhan dien hinh anh

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Load DNN model for face descriptor
descriptorModel = cv2.dnn.readNetFromTorch('openface.nn4.small2.v1.t7')

fontface = cv2.FONT_HERSHEY_SIMPLEX

model_path = 'face.tflite'
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


# Load the TensorFlow Lite model
interpreter = load_model(model_path)

def preprocess_input(frame):
    img = cv2.resize(frame, (120, 120))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    return img

def predict_identity(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)

            # Extract the face ROI and preprocess it
            faceROI = image[startY:endY, startX:endX]
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
            # Preprocess the input frame
            input_data = preprocess_input(faceROI)

            # Run inference
            output_data = predict(interpreter, input_data)
            result = np.argmax(output_data)
            labels = ["Dat", "Hung","Khanh", "Kien"] #, "Kem", "Dau"
    return labels[result]

def handle(msg):
    chat_id = msg['chat']['id']
    content_type, _, _ = telepot.glance(msg)
    if content_type == 'text' :
        user_input = msg['text']
        response = chatbot_response(user_input)
        bot.sendMessage(chat_id, response)
    elif content_type == 'photo':
        file_id = msg['photo'][-1]['file_id']
        file_path = bot.getFile(file_id)['file_path']
        image_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"

        response = requests.get(image_url)
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

        digit = predict_identity(image)

        bot.sendMessage(chat_id, f"The predicted face is: {digit}")

TOKEN = "6445196809:AAEXDf2vNeTAQDjRlURPKbV3JPtJj8hkFSU"
bot = telepot.Bot(TOKEN)

MessageLoop(bot, handle).run_as_thread()

print('Listening ...')

import time

while True:
    time.sleep(10)
