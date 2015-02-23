import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from PIL import Image 
from scipy import ndimage

from scipy import misc






import pylab

pic_dir = 'D:\Projects\LicensePlateRecognition\main\\'
file_name = 'res.jpg'
fp = open(pic_dir + file_name, 'rb')
image_file1 = Image.open(fp)
#plt.imshow(image_file1)
#plt.show()
image_file2=image_file1.convert('L')
pylab.gray()

#plt.imshow(image_file2)
#plt.show() # convert image to monochrome - this works
#image_file3= image_file2.convert('1')
#plt.imshow(image_file)
#plt.show()
print("sucess")
l=np.array(image_file1)
print(l)
# uses the Image module (PIL)

med_denoised = ndimage.median_filter(image_file2, 3)

misc.imsave('med_denoised.jpg', med_denoised)

img=mpimg.imread("med_denoised.jpg")
pylab.gray()
plt.imshow(l)
plt.show()


print("sucess")





"""

img=mpimg.imread('3.jpg')

 """       
a=np.array(img)
aa=a
print(len(a))
print(len(a[0]))

print(a[0])
b=a


pylab.gray()
#plt.imshow(a)
#plt.show()
c=0
a1=0
bw = np.empty((len(a),len(a[0])))
bw.fill(0)
for i in range(0,len(a)):
    for j in range(0,len(a[0])):
        if b[i][j]>150:
            c+=1
            bw[i][j]=255
        if b[i][j]<=150:
            a1+=1
            bw[i][j]=0
#plt.imshow(bw)
#plt.show()
print(bw)
def lineseg(bw):
    noofcoloums=len(bw[0])
    noofrows=len(bw)
    print(noofcoloums,noofrows)

    sumofrows=[]
    sum1=0
    k=sum(bw[0])/255
    k=np.int(k)
    print(k)

    for i in range(0,noofrows):

        k=sum(bw[i])/255
        k=np.int(k)
        sumofrows.append(k)
    print(len(sumofrows))
    print(sumofrows)
    for i in range(0,len(sumofrows)):
        if sumofrows[i]==1158:
            sum1+=1
    print(sum1)

    segment1=[]
    for i in range(0,len(sumofrows)-1):
        if sumofrows[i+1]<1158:
            segment1.append(i)
    print(segment1)

    segment=[]
    for i in range(0,len(sumofrows)-1):
        if sumofrows[i]==noofcoloums and sumofrows[i+1]<noofcoloums:
            segment.append(i-1)
        if sumofrows[i]<noofcoloums and sumofrows[i+1]==noofcoloums:
            segment.append(i+1)
    print(segment)
    line=[]
    k=0
    for l in range(0,len(segment)/2):
        x=0
        print(k)
        diff=segment[k+1]-segment[k]+2
        image1=np.empty((diff,noofcoloums))
        image1.fill(255)
        print(segment[k],segment[k+1])
        for i in range(segment[k],segment[k+1]):

            y=0

            for j in range(0,noofcoloums):
                image1[x][y]=bw[i][j]
                y=y+1
            x=x+1
        k=k+2

        line.append(image1)
    print(len(line))

    return line

#char

line=lineseg(bw)
line1=[]
for i1 in range(0,len(line)):
    sumofcols=[]
    sum1=0
    k=0
    a=line[i1]
    lena=len(a)
    lenac=len(a[i1])

    for i in range(0,lenac):
       for j in range(0,lena): 
           k+=a[j][i]
       k=k/255
       k=np.int(k)


       sumofcols.append(k)



    #print(len(sumofrows))
    print(lena)
    print(lenac)

    print(sumofcols)
    for i in range(0,len(sumofcols)):
        if sumofcols[i]==lenac:
            sum1+=1
    print(sum1)

    segment2=[]
    newsymbol=0

    print("valueeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    print(lena,lenac)
    for i in range(0,len(sumofcols)-1):
        if sumofcols[i]>lena-4  and (sumofcols[i+1])<=lena-4 and newsymbol==0:
            segment2.append(i)
            newsymbol=1
        if sumofcols[i]<lena-1 and newsymbol==1 and sumofcols[i+1]>lena-4:
            segment2.append(i+1)
            newsymbol=0
        print(i,sumofcols[i])
    print(lena,lenac)
    print("segment")
    print(segment2)
       #272, 298, 296, 311, 308, 323, 322, 336, 334, 345, 346, 361, 358, 373, 372, 397, 397, 411, 410, 437, 435, 449, 448, 462, 460, 474, 472, 486, 484, 498, 498, 511, 509, 523, 521, 535, 534, 547, 547, 560
    k=0
    for l in range(0,len(segment2)/2):
        x=0
        print(k)
        diff=segment2[k+1]-segment2[k]+2
        image1=np.empty((lena,diff))
        image1.fill(255)
        print(segment2[k],segment2[k+1])
        for i in range(segment2[k],segment2[k+1]):

            y=0

            for j in range(0,lena):
                image1[y][x]=a[j][i]
                y=y+1
            x=x+1
        k=k+2
        408, 435, 436, 464, 466, 493,
        494, 497, 505, 509, 534, 536, 563
        img1=lineseg(image1)
        line1.append(img1)

print(len(line1),len(line1[0][0]),len(img1))        
for i in range(0,len(line1)):      
    #plt.imshow(line1[i])
    #plt.show()
    misc.imsave("%s.jpg"%(i),line1[i][0])
    print(len(line1),len(line1[i]))
    print("end")



"""
    x=0   
    415, 441
    test=np.empty((150-115,445-415))
    test.fill(255)
    print(len(aa),len(aa[0]))
    print(len(test),len(test[0]))
    for i in range(115,150):

            y=0

            for j in range(415,445):
                test[x][y]=aa[i][j]
                y=y+1
            x=x+1


    misc.imsave('192.jpg',test)

    plt.imshow(test)
    plt.show()
"""