import ntpath

def check(data):
    def path_leaf(path):
        head,tail=ntpath.split(path)
        return tail#tail is everything after the final slash inshort to only extract file name
    data['center']=data['center'].apply(path_leaf)
    data['left']=data['left'].apply(path_leaf)
    data['right']=data['right'].apply(path_leaf)
    return data


import cv2
img=cv2.imread()



