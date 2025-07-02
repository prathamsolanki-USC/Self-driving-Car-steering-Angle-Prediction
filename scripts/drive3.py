import socketio
import speech_recognition as sr
sio = socketio.Client()

@sio.event
def connect():
    print('connection established!!')

# @sio.event
# def my_message(data):
#     print('message received with ', data)
#     sio.emit('my response', {'response': 'my response'})
#
# @sio.event
# def disconnect():
#     print('disconnected from server')

sio.connect('http://localhost:4567')

sio.emit("voice_command",{"commmand":"stop"})

r1 = sr.Recognizer()
# r2 = sr.Recognizer()
r3 = sr.Recognizer()
speed_limit2 = 10

while True:

    with sr.Microphone() as source:
        print('speak now')
        audio = r3.listen(source)
        # print(audio)

        try:
            if 'stop' in r1.recognize_google(audio):
                r1 = sr.Recognizer()
                speed_limit2 = 0.01
                sio.emit("voice_command", {"command": "stop"})
                print("Stopping the car")

            if 'start' in r1.recognize_google(audio):
                r1 = sr.Recognizer()
                speed_limit2 = 0.01
                # print(r1.recognize_google(audio))
                sio.emit("voice_command", {"command": "start"})
                print("Starting the car")


            if 'increase' in r1.recognize_google(audio):
                r1 = sr.Recognizer()
                speed_limit2 = 0.01
                # print(r1.recognize_google(audio))
                sio.emit("voice_command", {"command": "increase"})
                print("increasing the speed")


            if 'decrease' in r1.recognize_google(audio):
                r1 = sr.Recognizer()
                speed_limit2 = 0.01
                # print(r1.recognize_google(audio))
                sio.emit("voice_command", {"command": "decrease"})
                print("decreasing the speed")
        except:
            print("Could not recognise, say again")



