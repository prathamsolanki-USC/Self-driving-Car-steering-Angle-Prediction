from flask import Flask
import socketio
import eventlet
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import time
import random

increase=0
count = 0
sio = socketio.Server()#used to perform real time communication btw client and server.

app = Flask(__name__)  # __main__
speed_limit = 15
stop_car = False
speed_limit2=-1
def img_preprocess(img):
    # img=mpimg.imread(img)
    img = img[60:135:, :, ]  # removing hood of car from image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # this color is NVIDEA model
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


@sio.on('telemetry')  # working on image which will be captured while self driving
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))

    # x = random.randint(-10, 10)
    throttle = 1.0 - speed / speed_limit  # +x#if throttle zero then car will
    if not stop_car:
        if increase==1:
            send_control(steering_angle, throttle+0.3)

        if increase == -1:
            send_control(steering_angle, throttle - 0.3)
        else:
            send_control(steering_angle, throttle)

    else:
        send_control(steering_angle,0)



@sio.on('voice_command')
def voice_func(sid, data):
    global stop_car
    global increase

    if data["command"]=="stop":
        stop_car = True

    elif data["command"]=="start":
        stop_car=False

    elif data["command"]=="increase":
        increase = 1
        stop_car=False

    elif data["command"]=="decrease":
        increase = -1
        stop_car=False



@sio.on('connect')  # message,disconnect
def connect(sid, environ):
    print('Connected')



def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__(),

    })

if __name__ == '__main__':
    # app.run(port=3000)
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
