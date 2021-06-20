

import cgi
import cv2
import numpy
import base64
import os
import dlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from detector import DrowsinessDetector

class TestDetectionServer(BaseHTTPRequestHandler):
    detector = DrowsinessDetector()

    def __init__(self, request, client_address, server):
        print("Constructor called")
        BaseHTTPRequestHandler.__init__(self, request, client_address, server)

    def _set_headers(self, responseCode):
        self.send_response(responseCode)
        self.send_header("Content-type", "text/json")
        self.end_headers()

    def _html(self, message):

        return message.encode("utf8")  

    def do_GET(self):
        self._set_headers(200)
        self.wfile.write(self._html("SÜRÜCÜ YORGUNLUK TESPİTİ UYGULAMASINA HOŞGELDİNİZ"))

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        print('Post!')

        response = {"drowsy": False}

        tmpFile = open("tmp.mpeg", 'wb')
        tmpFile.write(self.rfile.read(int(self.headers["content-length"])))
        print("Bytes read: ", self.headers["content-length"])

        tmpFile.close()
        vidObj = cv2.VideoCapture("tmp.mpeg")
        count = 0
        success = 1

        errored = False

        while success:
            success, image = vidObj.read()
            if success != 1:
                break

            if count % 5 == 0:
                try:
                    cv2.imwrite("tmpFrame.jpg", image)
                    img = dlib.load_grayscale_image("tmpFrame.jpg")
                    appearsDrowsy = TestDetectionServer.detector.areEyesClosed(img)

                    print(appearsDrowsy)
                    print("current consecutive drowsy frames: ", TestDetectionServer.detector.getNumberConsecutiveDrowsyFrames())
                    if appearsDrowsy != None:
                        if not appearsDrowsy:
                            TestDetectionServer.detector.resetNumberConsecutiveDrowsyFrames()

                        if(TestDetectionServer.detector.isDrowsy()):
                            response["drowsy"] = True
                            break

                    if os.path.exists("tmpFrame.jpg"):
                        os.remove("tmpFrame.jpg")

                except Exception as e:
                    errored = True
                    print(e)
                    break

            count += 1
        print(count)


        if errored:
            self._set_headers(500)
        else:
            self._set_headers(200)

        print(response)
        self.wfile.write(self._html(str(response)))

def run(server_class=HTTPServer, handler_class=TestDetectionServer, addr="localhost", port=8000):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

    print(f"Sunucu başlatıldı {addr}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
	run(addr = "", port = 8000)
