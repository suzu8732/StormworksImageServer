import os
import glob
import cv2
import numpy as np
from http.server import HTTPServer
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
from urllib.parse import parse_qs
from http import HTTPStatus

IMAGE_EXTENTIONS = ('jpg', 'png')
PORT = 8080
GAMMA = 0.45
ALPHA = 0.75
BETA = 0.9

file_names = []
images = {}

class ImageDataRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        monitor_width = int(params['width'][0])
        monitor_hight = int(params['hight'][0])
        file_name = params['fileName'][0]
        start = int(params['start'][0])
        num = int(params['num'][0])

        self.send_response(HTTPStatus.OK)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        
        if file_name in file_names:
            response_data = get_response_data(file_name, monitor_width, monitor_hight, num, start)
            self.wfile.write(response_data.encode())
        else:
            self.wfile.write('NOT_FOUND'.encode())

def create_lut():
    lut = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        lut[i][0] = 255 * (float(i) / 255) ** (1.0 / GAMMA)
    return lut

def loading_images():
    global file_names, images

    print('Start loading')
    for ext in IMAGE_EXTENTIONS:
        file_path = f'image/*.{ext}'
        names = [os.path.split(name)[1] for name in glob.glob(file_path)]
        file_names.extend(names)


    lut = create_lut()
    images = {name: cv2.convertScaleAbs(cv2.LUT(cv2.imread(f'image/{name}'), lut),alpha=ALPHA, beta=BETA) for name in file_names}
    
    print('Filelist')
    for name in file_names:
        print(f' ・{name}')

def resize_image(image: cv2.typing.MatLike, width, hight):
    origin_hight, origin_width = image.shape[:2]

    if origin_hight > origin_width:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        origin_hight, origin_width = image.shape[:2]

    original_ratio = origin_width / origin_hight
    new_ratio = width / hight
    if new_ratio < original_ratio:
        center_width = origin_width / 2
        crop_width = origin_hight * new_ratio
        image = image[0 : origin_hight, int(center_width - crop_width / 2) : int(center_width + crop_width // 2)]
    else:
        center_hight = origin_hight / 2
        crop_hight = origin_width / new_ratio
        image = image[int(center_hight - crop_hight / 2) : int(center_hight + crop_hight / 2), 0 : origin_width]

    image = cv2.resize(image, (width, hight))
    
    return image

def get_lines_data(image, num, start, monitor_hight):
    lines_data = ['OK&']
    for line in range(start, start + num):
        if line > monitor_hight:
            break

        line_data = [f'{pixel[2]}:{pixel[1]}:{pixel[0]}' for pixel in image[line - 1]]
        lines_data.append(','.join(line_data))

    return ';'.join(lines_data)

def get_response_data(file_name, monitor_width, monitor_hight, num, start):
    image = images.get(file_name)
    resized_image = resize_image(image, monitor_width, monitor_hight)
    lines_data = get_lines_data(resized_image, num, start, monitor_hight)

    return lines_data
    
def serve_image():
    print('Start server')
    server = HTTPServer(('localhost', PORT), ImageDataRequestHandler)
    server.serve_forever()

def main():
    loading_images()
    serve_image()

if __name__ == '__main__':
    main()
