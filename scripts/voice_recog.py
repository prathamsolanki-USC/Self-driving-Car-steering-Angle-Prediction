import speech_recognition as sr
import webbrowser as wb

r1=sr.Recognizer()
r2=sr.Recognizer()
r3=sr.Recognizer()
speed_limit2=10

with sr.Microphone() as source:
    print('[search edureka:search youtube]')
    print('speak now')
    audio=r3.listen(source)
    # print(audio)

if 'stop' in r1.recognize_google(audio):
    r1=sr.Recognizer()
    speed_limit2=0.01
    print(r1.recognize_google(audio))






    # url='https://www.youtube.com/'

    # with sr.Microphone() as source:
    #     print('search your query')
    #     audio=r1.listen(source)
    #
    #     try:
    #         get=r1.recognize_google(audio)
    #         print(get)
    #
    #     except sr.UnknownValueError:
    #         print('error')
    #     except sr.RequestError as e:
    #         print('failed'.format(e))

