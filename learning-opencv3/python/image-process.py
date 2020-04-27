import cv2

img_path = '../resources/tmdyh.jpg'


def show_image(image):
    cv2.imshow(winname='show the image', mat=image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def get_image_info(image):
    print("图片大小:", image.shape)
    pix_bgr = image[100, 100]
    # print(pix_bgr)
    # print(image[100, 100, 0])
    # print(image[100, 100, 1])
    # print(image[100, 100, 2])
    # print(image.dtype)
    # print(cv2.GetSize(image))


def resize(image):
    r = 100.0 / image.shape[1]
    dim = (100, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    print(resized.shape)
    cv2.imshow("resized", resized)
    # cv2.waitKey(0)


def crop(image):
    (h, w) = image.shape[:2]
    crop1 = image[0:int(h / 3), 0:w]
    crop2 = image[int(h / 3):int(2 * h / 3), 0:w]
    crop3 = image[int(2 * h / 3):h, 0:w]
    cv2.imshow("crop1", crop1)
    cv2.imshow("crop2", crop2)
    cv2.imshow("crop3", crop3)
    path = '/Volumes/develop/code-repository/practice-code/learning-opencv3/resources/'

    cv2.imwrite(path+"1.jpg", crop1)
    cv2.imwrite(path+"2.jpg", crop2)
    cv2.imwrite(path+"3.jpg", crop3)

def puttext(image):
    x, y = 100, 100
    cv2.putText(img=image, text="kuaikanmanhua", org=(x, y), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1, color=(255, 255, 255))
    cv2.imshow(winname='add text  on image', mat=image)


def addImage(image):
    path = '/Volumes/develop/code-repository/practice-code/learning-opencv3/resources/'
    matimage = image

    # matimagenew = np.zeros((matimage.shape[0],matimage.shape[1],3))
    matimagenew = matimage - matimage
    watermark_template_filename = path + 'logo.png'
    matlogo = cv2.imread(watermark_template_filename)

    matimagenew[matimage.shape[0] - matlogo.shape[0]:matimage.shape[0],
        matimage.shape[1] - matlogo.shape[1]:matimage.shape[1]] = matlogo
    imagenew = cv2.addWeighted(matimage, 1, matimagenew, 1, 1)
    savepath = path + 'addLogo.png'
    cv2.imshow("addlogo", imagenew)
    # cv2.imwrite(savepath, imagenew)


if __name__ == '__main__':
    image = cv2.imread(filename=img_path)

    show_image(image)
    get_image_info(image)
    resize(image)
    crop(image)
    puttext(image)
    addImage(image)
    cv2.waitKey(0)
    cv2.dft()
