import cv2

cap = cv2.VideoCapture('C:\\Users\\Lukas\\Documents\\Python Projects\\TestData\\cats.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(frame,(32,32),interpolation = cv2.INTER_AREA)
    
    cv2.imshow('frame',img)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()