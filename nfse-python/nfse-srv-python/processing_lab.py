import numpy as np
import imutils
import cv2
from PIL import Image, ImageColor,ImageFilter, ImageEnhance
import pytesseract
from collections import Counter
import requests
import tensorflow as tf
import pytesseract

# 1 = caxias
# 2 = barueri
# 3 = niteroi
# 7 = aparecido de goiania
# 8 = belo horizonte
# 9 = blumenau

'''
model1 = 1,2
model2 = 7
model3 = 8
model4 = SJ,Criciuma,Imbituba
model5 = 9
'''
def model1(captcha):
    """
       A function to create contours of captcha's letters for cities 1 and 2
       :param captcha: raw captcha image
       :return: array with letter images
    """

    img = cv2.cvtColor(captcha, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("Output", img)
    # cv2.waitKey()
    _, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)
    thresh = cv2.bitwise_not(thresh)
    resized = cv2.resize(thresh, (140,60), interpolation=cv2.INTER_AREA)
    # cinza
    # img_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.copyMakeBorder(resized, 10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
    #Area = (80,160) after border
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # preto e branco
    _, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)

    # encontrar contornos de letras
    contornos, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    regiao_letras = []

    # filtrar contornos que sao realmente letras
    max_w, max_h = img.shape[:2]
    for contorno in contornos:
        (x, y, l, a) = cv2.boundingRect(contorno)
        if l >= max_w:
            pass
        else:
            area = cv2.contourArea(contorno)
            # print('Area:', area)
            # print(l, a)
            #Barueri: 120; Caxias:155
            if area > 110:
                if l / a > 1:
                    half_width = int(a / 2)
                    regiao_letras.append((x, y, half_width, a))
                    regiao_letras.append((x + half_width, y, half_width, a))
                else:
                    regiao_letras.append((x, y, l, a))

    regiao_letras = sorted(regiao_letras, key=lambda x: x[0])

    #remover menor area caso reconheça uma letra a mais
    prod = []
    if len(regiao_letras) > 5:
        for i in regiao_letras:
            prod.append(i[2] * i[3])
        min_index = prod.index(min(prod))
        del regiao_letras[min_index]

    # desenhar contornos e separar em arquivos
    img_final = cv2.merge([img] * 3)
    lista_letras = []
    for retangulo in regiao_letras:
        x, y, l, a = retangulo
        img_letra = img[y - 2: y + a + 2, x - 2: x + l + 2]

        cv2.rectangle(img_final, (x - 2, y - 2), (x + l + 2, y + a + 2), (0, 255, 0), 1)
        lista_letras.append(img_letra)
        # cv2.imshow("Output", img_letra)
        # cv2.waitKey()
    return lista_letras

def model2(captcha):
    """
       A function to create contours of captcha's letters for city 7
       :param captcha: raw captcha image
       :return: array with letter images
    """
    img = cv2.cvtColor(captcha, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Output", img)
    # cv2.waitKey()
    img = img[15:55, 10:100]  # crop image to get dominant color

    dominant = img.copy()  # copy image
    # get dominant color to get text color
    unique, counts = np.unique(dominant.reshape(-1, dominant.shape[-1]), axis=0, return_counts=True)
    dominant[:, :, 0], dominant[:, :, 1], dominant[:, :, 2] = unique[np.argmax(counts)]
    dominant = list(dominant[0, 0])

    height, width, _ = img.shape
    '''
    colors in BGR code
    [blue,green,red]
    vermelho = [0,0,255]
    branco = [255,255,255]
    preto = [0,0,0]
    azul_claro = [255,0,0]
    azul_escuro = [139,0,0]
    '''

    bgr_codes = [[0, 0, 255], [255, 255, 255], [0, 0, 0], [255, 0, 0], [139, 0, 0]]

    for i in range(height):
        for j in range(width):
            # img[i,j] is the RGB pixel at position (i, j)
            if dominant == [255, 255, 255]:
                if any(img[i, j] != bgr_codes[bgr_codes.index(dominant)]):
                    img[i, j] = [255, 255, 255]
                elif any(img[i, j] == bgr_codes[bgr_codes.index(dominant)]):
                    img[i, j] = [0, 0, 0]
            elif any(img[i, j] != bgr_codes[bgr_codes.index(dominant)]):
                img[i, j] = [255, 255, 255]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    _, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)

    # resizing
    img_gray = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # reforcing image
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    blur2 = cv2.GaussianBlur(blur, (5, 5), 0)
    thresh2 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 3)

    kernel = np.ones((2, 1), np.uint8)
    img = cv2.dilate(thresh2, kernel, 1)
    # cv2.imshow("Output", img)
    # cv2.waitKey()
    # find letter contours
    contornos, heirar = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    regiao_letras = []

    # filter contours that are letters
    max_w, max_h = img.shape[:2]
    for contorno in contornos:
        (x, y, l, a) = cv2.boundingRect(contorno)
        if l >= max_w:
            pass
        else:
            area = cv2.contourArea(contorno)
            # print('Area:', area)
            # print(l, a)
            if area >= 125:
                if l / a > 1:
                    half_width = int(a / 2)
                    regiao_letras.append((x, y, half_width, a))
                    regiao_letras.append((x + half_width, y, half_width, a))
                else:
                    regiao_letras.append((x, y, l, a))

    regiao_letras = sorted(regiao_letras, key=lambda x: x[0])
    prod = []
    if len(regiao_letras) > 4:
        for i in regiao_letras:
            prod.append(i[2] * i[3])
        min_index = prod.index(min(prod))
        del regiao_letras[min_index]

    # desenhar contornos e separar em arquivos
    img_final = cv2.merge([img] * 3)
    lista_letras = []
    for retangulo in regiao_letras:
        x, y, l, a = retangulo
        img_letra = img[y - 2: y + a + 2, x - 2: x + l + 2]

        cv2.rectangle(img_final, (x - 2, y - 2), (x + l + 2, y + a + 2), (0, 255, 0), 1)
        lista_letras.append(img_letra)
        # cv2.imshow("Output", img_letra)
        # cv2.waitKey()
    return lista_letras

def model3(captcha):
    """
       A function to create contours of captcha's letters for city 8
       :param captcha: raw captcha image
       :return: array with letter images
    """

    img = cv2.cvtColor(captcha, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Output", img)
    # cv2.waitKey()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("Output", img)
    # cv2.waitKey()
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)


    thresh = cv2.bitwise_not(thresh)
    # cv2.imshow("Output", thresh)
    # cv2.waitKey()

    img = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # cv2.imshow("Output", img)
    # cv2.waitKey()

    # threshold
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
    # cv2.imshow("Output", img)
    # cv2.waitKey()

    # find letter contours
    contornos, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    regiao_letras = []

    # filter contours that are letters
    max_w, max_h = img.shape[:2]

    for contorno in contornos:
        (x, y, l, a) = cv2.boundingRect(contorno)
        if l >= max_w:
            pass
        else:
            area = cv2.contourArea(contorno)
            # print('Area:', area)
            # print(l, a)
            if area >= 80:
                if l / a > 1.1:
                    half_width = int(a / 2)
                    regiao_letras.append((x, y, half_width, a))
                    regiao_letras.append((x + half_width, y, half_width, a))
                else:
                    regiao_letras.append((x, y, l, a))

    regiao_letras = sorted(regiao_letras, key=lambda x: x[0])
    # print(regiao_letras)
    prod = []
    if len(regiao_letras) > 5:
        for i in regiao_letras:
            prod.append(i[2] * i[3])
        min_index = prod.index(min(prod))
        del regiao_letras[min_index]

    # desenhar contornos e separar em arquivos
    img_final = cv2.merge([img] * 3)
    lista_letras = []
    for retangulo in regiao_letras:
        x, y, l, a = retangulo
        img_letra = img[y - 2: y + a + 2, x - 2: x + l + 2]

        cv2.rectangle(img_final, (x - 2, y - 2), (x + l + 2, y + a + 2), (0, 255, 0), 1)
        lista_letras.append(img_letra)
        # cv2.imshow("Output", img_letra)
        # cv2.waitKey()
    return lista_letras

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]
    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)
    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)
    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    return image

def classify_captcha(captcha):
    img = cv2.cvtColor(captcha, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def model4(url_code,attempts):
    # Initialize letter positions
    pos1 = list()
    pos2 = list()
    pos3 = list()
    pos4 = list()
    pos5 = list()
    pos6 = list()
    tries = 1
    while tries <= attempts:
        # get new captcha image
        img_url = "https://e-gov.betha.com.br/e-nota/bfcfaces/captcha.jpg?text=" + url_code

        img_data = requests.get(img_url).content
        nparr = np.fromstring(img_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Open image with opencv
        image = np.asarray(img_np)

        #Load image to PIL
        imageObject = Image.fromarray(image)

        # Size of the image in pixels (size of original image)
        width, height = imageObject.size

        # Setting size of each image
        size = width / 6
        size_ex = size * 1.2

        left = 0
        count = 1

        while count <= 6:

            top = 1
            bottom = height

            if count == 1:
                left = 0
            else:
                left = left + size

            right = left + size_ex

            img = imageObject.crop(((left * 0.9), top, right, bottom))

            img.filter(ImageFilter.SHARPEN)
            img.filter(ImageFilter.SHARPEN)
            img.filter(ImageFilter.SHARPEN)

            text = pytesseract.image_to_string(img, lang="eng", config='--psm 10')

            caracter = ""
            for char in text:
                if char.isalpha() == True:
                    if char.isupper() == True:
                        caracter = char

            if caracter != "":
                if count == 1: pos1.append(caracter)
                if count == 2: pos2.append(caracter)
                if count == 3: pos3.append(caracter)
                if count == 4: pos4.append(caracter)
                if count == 5: pos5.append(caracter)
                if count == 6: pos6.append(caracter)

            count = count + 1 #increase letter position
        tries = tries + 1 #increase try count

    captcha = str(Counter(pos1).most_common()[0][0] + Counter(pos2).most_common()[0][0] +
                  Counter(pos3).most_common()[0][0] + Counter(pos4).most_common()[0][0] +
                  Counter(pos5).most_common()[0][0] + Counter(pos6).most_common()[0][0])
    return captcha

def model5(img,img_height,img_width):
    # Original image
    # raw_img = cv2.imread(file)
    # convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # apply median blur
    blur = cv2.medianBlur(img, 3)
    # treshold
    _, img = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # reduce line thickness
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.imencode('.png', img)[1].tostring() #encode image to png
    # 1. Read image
    # img = tf.io.read_file(file)
    # img = tf.convert_to_tensor(img, dtype=tf.float32)
    # print('SHAPE0', img.shape)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # print('SHAPE1', img.shape)
    # img = tf.expand_dims(img, 2)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # print('SHAPE2', img.shape)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # print('SHAPE3', img.shape)
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # print('SHAPE4', img.shape)
    img = np.expand_dims(img, axis=0)

    return img