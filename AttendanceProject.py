import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


lokasi = 'ImageAttendence'
gambar = []
NamaKelas = []
List = os.listdir(lokasi)
print(List)
for nk in List:
    LokGambar = cv2.imread(f'{lokasi}/{nk}')
    gambar.append(LokGambar)
    NamaKelas.append(os.path.splitext(nk)[0])
print(NamaKelas)

def myCode(gambar):
    listcodeku = []
    for img in gambar:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kode = face_recognition.face_encodings(img)[0]
        listcodeku.append(kode)
    return listcodeku

def folder(name):
    with open('Attendance.csv','r+')as f:
        listData = f.readlines()
        listNama = []
        for line in listData:
            entry = line.split(',')
            listNama.append(entry[0])
        if name not in listNama:
            now = datetime.now()
            jam = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{jam}')
            #today = date.today()
            #dString = today.replace('%Y/%M/%D')
            #f.writelines(f'\n{name},{dString}')




listcodekuKnown = myCode(gambar)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    tampilanMuka = face_recognition.face_locations(imgS)
    FrameKode = face_recognition.face_encodings(imgS,tampilanMuka)

    for kodeMuka,faceLoc in zip(FrameKode,tampilanMuka):
        perbandingan = face_recognition.compare_faces(listcodekuKnown,kodeMuka)
        tampilWajah = face_recognition.face_distance(listcodekuKnown,kodeMuka)
        #print(faceDis)
        indexPerbandingan = np.argmin(tampilWajah)

        if perbandingan[indexPerbandingan]:
            name = NamaKelas[indexPerbandingan].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            folder(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)





# faceLoc = face_recognition.face_locations(imgJeef)[0]
# encodeJeef = face_recognition.face_encodings(imgJeef)[0]
# cv2.rectangle(imgJeef,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# faceLocBenzof = face_recognition.face_locations(imgBenzof)[0]
# encodeBenzof = face_recognition.face_encodings(imgBenzof)[0]
# cv2.rectangle(imgBenzof,(faceLocBenzof[3],faceLocBenzof[0]),(faceLocBenzof[1],faceLocBenzof[2]),(255,0,255),2)

# results = face_recognition.compare_faces([encodeJeef],encodeBenzof)
# faceDis = face_recognition.face_distance([encodeJeef],encodeBenzof)