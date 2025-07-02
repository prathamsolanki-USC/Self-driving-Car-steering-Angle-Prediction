from flask import Flask
import socketio
import eventlet
sio=socketio.Server()
import speech_recognition as sr
import webbrowser as wb



app=Flask(__name__)#__main__

r1 = sr.Recognizer()
r2 = sr.Recognizer()
r3 = sr.Recognizer()
speed_limit2 = 10

with sr.Microphone() as source:
    print('[search edureka:search youtube]')
    print('speak now')
    audio = r3.listen(source)
    # print(audio)

if 'stop' in r1.recognize_google(audio):
    r1 = sr.Recognizer()
    speed_limit2 = 0.01
    print(r1.recognize_google(audio))
   
@sio.on('connect')#message,disconnect
def connect(sid,environ):
    # print(sid, " ", environ)
    print('Connected file2')
    send_control(0,10)

def send_control(steering_angle,throttle):
    sio.emit('steer',data={
        'steering_angle':steering_angle.__str__(),
        'throttle':throttle.__str__()

    })
# @app.route('/')
# def greeting():
#     return 'Welcome!'

if __name__=='__main__':
    # app.run(port=3000)
    # model=load_model('model.h5')
    app=socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(('',4567)),app)