import socketio
import gunicorn
import fcntl
sio=socketio.Server()
app=socketio.WSGIApp(sio)


@sio.event
def connect(sid,environ):#session id,
    print(sid,'connected')
@sio.event
def disconnect(sid):
    print(sid ,'disconnected')


