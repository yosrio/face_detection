import numpy as numpy
import cv2 as cv

# Ambil data wajah
face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# Baca data gambar
image = cv.imread('../data/data_3.jpg')

# Mengubah ukuran gambar agar tidak terlalu besar
scale_percent = 60
width = int(image.shape[1] * scale_percent / 120)
height = int(image.shape[0] * scale_percent / 120)
dim = (width, height)
image_resize = cv.resize(image, dim, interpolation = cv.INTER_AREA)

# Mengubah gambar resize menjadi abu-abu
image_gray = cv.cvtColor(image_resize, cv.COLOR_BGR2GRAY)

# Fungsi dari library openCV untuk mendeteksi wajah
face_recognition = face.detectMultiScale(image_gray, 1.1, 5)
# Membuat type font
font = cv.FONT_HERSHEY_SIMPLEX
# Variabel untuk menampung jumlah wajah
sum2 = 0

# Perulangan sebanyak wajah yang terdeteksi
for(x,y,w,h) in face_recognition:
    # Menjumlahkan banyak wajah
    sum2 =  sum2 + 1
    # Menampilkan text pada wajah
    cv.putText(image_resize, "Face", (x, y-10), font, 0.75, (0, 0, 255), 2, cv.LINE_AA)
    # Menampilkan kotak sebagai penanda wajah yang terdeteksi
    cv.rectangle(image_resize, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = image_gray[y:y+h, x:x+w]
    roi_color = image_resize[y:y+h, x:x+w]

# Menampilkan text untuk jumlah wajah terdeteksi
cv.putText(image_resize, "Jumlah wajah ada : "+ str(sum2) + " buah", (10,30), font, 1, (0,0,0), 2, cv.LINE_AA)
# Menampilkan gambar yang sudah di resize
cv.imshow('image', image_resize)
cv.waitKey(0)
cv.destroyAllWindows()
