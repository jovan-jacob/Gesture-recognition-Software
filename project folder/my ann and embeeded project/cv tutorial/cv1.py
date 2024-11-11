import cv2
# image = cv2.imread("D:\courses\python\my ann and embeeded project\image.jpg")
# cv2.imshow('Image', image)
# cv2.waitKey(0)  # Waits indefinitely until a key is pressed
# cv2.destroyAllWindows()

# cv2.imwrite('D:\courses\python\my ann and embeeded project\cv tutorial\save_image.jpg', image)
cap = cv2.VideoCapture(0)  # 0 is the default camera
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Camera Feed', frame)
    input()
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()