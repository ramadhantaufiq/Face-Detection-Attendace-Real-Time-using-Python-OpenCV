import cv2
import numpy as np
import face_recognition
from datetime import date

imgJeef = face_recognition.load_image_file('ImageBasic/Jeef.jpg')
imgJeef = cv2.cvtColor(imgJeef,cv2.COLOR_BGR2RGB)
imgBenzof = face_recognition.load_image_file('ImageBasic/Dedi.jpg')
imgBenzof = cv2.cvtColor(imgBenzof,cv2.COLOR_BGR2RGB)
imgTaufiq = face_recognition.load_image_file('ImageBasic/Taufiq.jpg')
imgTaufiq = cv2.cvtColor(imgTaufiq,cv2.COLOR_BGR2RGB)
imgFia = face_recognition.load_image_file('ImageBasic/Fia.jpg')
imgFia = cv2.cvtColor(imgFia,cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgJeef)[0]
encodeJeef = face_recognition.face_encodings(imgJeef)[0]
cv2.rectangle(imgJeef,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocBenzof = face_recognition.face_locations(imgBenzof)[0]
encodeBenzof = face_recognition.face_encodings(imgBenzof)[0]
cv2.rectangle(imgBenzof,(faceLocBenzof[3],faceLocBenzof[0]),(faceLocBenzof[1],faceLocBenzof[2]),(255,0,255),2)

faceLocTaufiq = face_recognition.face_locations(imgTaufiq)[0]
encodeTaufiq = face_recognition.face_encodings(imgTaufiq)[0]
cv2.rectangle(imgTaufiq,(faceLocTaufiq[3],faceLocTaufiq[0]),(faceLocTaufiq[1],faceLocTaufiq[2]),(255,0,255),2)

faceLocFia = face_recognition.face_locations(imgFia)[0]
encodeFia = face_recognition.face_encodings(imgFia)[0]
cv2.rectangle(imgFia,(faceLocFia[3],faceLocFia[0]),(faceLocFia[1],faceLocFia[2]),(255,0,255),2)

faceLocAsep = face_recognition.face_locations(imgAsep)[0]
encodeAsep = face_recognition.face_encodings(imgAsep)[0]
cv2.rectangle(imgAsep,(faceLocAsep[3],faceLocAsep[0]),(faceLocAsep[1],faceLocAsep[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeJeef],encodeBenzof)
faceDis = face_recognition.face_distance([encodeJeef],encodeBenzof)
print(results,faceDis)
cv2.putText(imgBenzof,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

results = face_recognition.compare_faces([encodeJeef],encodeTaufiq)
faceDis = face_recognition.face_distance([encodeJeef],encodeTaufiq)
print(results,faceDis)
cv2.putText(imgTaufiq,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Jeef',imgJeef)
cv2.imshow('Benzof',imgBenzof)
cv2.imshow('Taufiq',imgTaufiq)
cv2.imshow('Fia',imgFia)
cv2.imshow('Asep',imgAsep)
cv2.waitKey(0)