import cv2


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)  # Sử dụng camera USB

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if success:
            return image
        else:
            return None
