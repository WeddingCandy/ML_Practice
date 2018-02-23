#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image

if __name__ == '__main__':
    image_file = 'shan2.jpeg'
    height = 100

    img = Image.open(image_file)
    print('img = ',img)
    img_width, img_height = img.size
    width = int(2 * height * img_width // img_height)    # 假定字符的高度是宽度的2倍
    img = img.resize((width, height), Image.ANTIALIAS)
    pixels = np.array(img.convert('L'))
    print('type = ',type(pixels))
    print(pixels.shape)
    print(pixels)
    chars = "MNHQ$OC?7>!:-;. "
    N = len(chars)
    step = 256 // N
    print(N)
    result = ''
    for i in range(height):
        for j in range(width):
            result += chars[pixels[i][j] // step]
        result += '\n'
    with open('text3.txt', mode='w') as f:
        f.write(result)
