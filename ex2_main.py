from ex2_utils import *
import matplotlib.pyplot as plt
import time
import cv2
pics =[]

def conv2Demo(img):
    kerne= np.ones((5, 5))
    kerne/= kerne.sum()
    kernel = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
    cv2_img = cv2.filter2D(img, -1, kerne, borderType=cv2.BORDER_REPLICATE)
    st= time.time()
    con2D = conv2D(img, kernel)
    print("Time for test conv2D: %.2f" % (time.time()-st))
    f, ax = plt.subplots(1, 2)
    plt.gray()
    ax[0].imshow(cv2_img)
    ax[0].set_title("conv2d cv2")
    ax[1].imshow(con2D)
    ax[1].set_title("my implementation")
    plt.show()


def conv1Demo():
    kernel = [0, 2, 0]
    a = [1, 2, 3, 4, 5]
    test = conv1D(np.array(a), np.array(kernel))
    print("conv1D Results:", test)


def derivDemo(img):
    st = time.time()
    directions, magnitude, x_der, y_der = convDerivative(img)
    print("Time:%.2f" % (time.time() - st))
    plt.title("convDerivative")
    f, ax = plt.subplots(2, 2)
    plt.gray()
    ax[0][0].imshow(directions)
    ax[0][0].set_title("directions")
    ax[0][1].imshow(magnitude)
    ax[0][1].set_title("magnitude")
    ax[1][0].imshow(x_der)
    ax[1][0].set_title("x_der")
    ax[1][1].imshow(y_der)
    ax[1][1].set_title("y_der")
    plt.show()


def blurDemo(img:np.ndarray, kernel:int):
    st= time.time()
    img1= blurImage1(img, kernel)
    print("Time:%.2f" % (time.time() - st))
    st = time.time()
    img2 = blurImage2(img, kernel)
    print("Time:%.2f" % (time.time() - st))
    f, ax = plt.subplots(1, 2)
    plt.gray()
    ax[0].imshow(img1)
    ax[0].set_title("My Implementation")
    ax[1].imshow(img2)
    ax[1].set_title("openCV Implementation")
    plt.show()


def edgeDemo(img: np.ndarray, thresh:float, t1:float, t2:float):
    print("####### Sobel Method  #######")
    st = time.time()
    cvImg, myImg = edgeDetectionSobel(img, thresh)
    print("Time:%.2f" % (time.time() - st))
    f, ax = plt.subplots(1, 2)
    plt.gray()
    ax[0].imshow(myImg)
    ax[0].set_title("My Implementation")
    ax[1].imshow(cvImg)
    ax[1].set_title("openCV Implementation")
    plt.show()
    #cance\le?


    print("####### ZeroCrossing Method  #######")
    st = time.time()
    myImg = edgeDetectionZeroCrossingSimple(img)
    print("Time:%.2f" % (time.time() - st))
    plt.imshow(myImg)
    plt.title("ZeroCrossing Implementation")
    plt.show()

    print("####### Canny Edge Method  #######")
    st = time.time()
    cvImg, myImg = edgeDetectionCanny(img, t1, t2)
    #plt.figure?
    print("Time:%.2f" % (time.time() - st))
    f, ax = plt.subplots(1, 2)
    plt.gray()
    ax[0].imshow(myImg)
    ax[0].set_title("My Implementation")
    ax[1].imshow(cvImg)
    ax[1].set_title("openCV Implementation")
    plt.show()


def houghDemo(img, min_radius, max_radius):
    # pic = 'circles.jpg'
    #img= cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    st = time.time()
    res = houghCircle(img, min_radius, max_radius)
    print("Time:%.2f" % (time.time() - st))

    fig = plt.figure()
    plt.imshow(img)
    circles = []
    for r, x, y in res:
        circles.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
        fig.add_subplot().add_artist(circles[-1])
    plt.title("houghCircles")
    plt.show()


def main():
    #print ID
    print("ID:", myID())
    pics = []
    picH=[]
    sobelT=0.0
    cannyT1=0.0
    cannyT2=0.0
    pics.append('beach.jpg')
    pics.append('boxman.jpg')
    print(pics)
    picH.append('small-circles.jpg')
    picH.append('circles.jpg')
    conv1Demo()
    for pic in pics:
        img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        if pic== 'beach.jpg' :
            sobelT=0.6
            cannyT1= 0.4
            cannyT2= 0.7
        if pic == 'boxman.jpg' :
            sobelT = 0.4
            cannyT1 = 0.3
            cannyT2 = 0.42
        conv1Demo()
        conv2Demo(img)
        derivDemo(img)
    # # #try to change kernal size?
        blurDemo(img, kernel=5)
        edgeDemo(img , thresh=sobelT, t1=cannyT1, t2=cannyT2)
    for pi in picH:
        if  pi == 'circles.jpg':
            img2 = cv2.imread(pi, cv2.IMREAD_GRAYSCALE)
            houghDemo(img2 , min_radius=50, max_radius=65)
        if pi == 'small-circles.jpg':
            img2 = cv2.imread(pi, cv2.IMREAD_GRAYSCALE)
            houghDemo(img2 , min_radius=10, max_radius=15)
        if pi == 'small-circles.jpg':
            img2 = cv2.imread(pi, cv2.IMREAD_GRAYSCALE)
            houghDemo(img2, min_radius=10, max_radius=15)


if __name__ == '__main__':
    main()
