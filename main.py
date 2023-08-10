import face_recognition,cv2,numpy, csv
from datetime import  datetime
name=0
video_capture=cv2.VideoCapture(0)

#Load known faces
peter_photo=face_recognition.load_image_file("peter.jpg")
ben_photo=face_recognition.load_image_file("ben.jpg")
peter_encoding=face_recognition.face_encodings(peter_photo)[0]
ben_encoding=face_recognition.face_encodings(ben_photo)[0]

known_face_encodings=[peter_encoding,ben_encoding]
known_face_names=["Peter Parker","Ben Affleck"]

#List of expected students
students=known_face_names.copy()
face_locations=[]
face_encodings=[]

#Get the current date and time of attendance
now=datetime.now()
current_date=now.strftime("%d-%m-%Y")

f=open(f"{current_date}.csv", "w+", newline="")
lnwriter=csv.writer(f)

while True:
   _, frame=video_capture.read()
   small_frame=cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
   rgb_small_frame=cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

   #Reconize faces
   face_locations=face_recognition.face_locations(rgb_small_frame)
   face_encodings=face_recognition.face_encodings(rgb_small_frame, face_locations)

   for face_encoding in face_encodings:
      matches=face_recognition.compare_faces(known_face_encodings, face_encoding)
      face_distance=face_recognition.face_distance(known_face_encodings,face_encoding)
      best_match_index=numpy.argmin(face_distance)

      if(matches[best_match_index]):
         name=known_face_names[best_match_index]

      if name in known_face_names:
         font=cv2.FONT_HERSHEY_TRIPLEX
         bottomLeftCornerOfText=(10,100)
         fontScale=1.5
         fontColor=(0,255,0)
         thickness=3
         lineType=2
         cv2.putText(frame,name+ " Present", bottomLeftCornerOfText, font,fontScale,fontColor,thickness,lineType)
      if name in students:
         students.remove(name)
         current_time=now.strftime("%H-%M%S")
         lnwriter.writerow([name,current_time])
      cv2.imshow("Attendance", frame)
      if cv2.waitKey(1) & 0xFF == ord("q"):
         break
video_capture.release()
cv2.destroyAllWindows()
f.close()
