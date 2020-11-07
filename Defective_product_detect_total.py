import cv2
import numpy as np
from math import *

img = cv2.imread("img_21.jpg", cv2.IMREAD_COLOR)
img_cp=img.copy()

arr_cir=[0,0,0,0,0,0]
arr_def=[0,0,0,0,0,0]
arr_color = [0,0,0,0,0,0]
arr_color_eng = [0,0,0,0,0,0]
arr_updn = [0,0,0,0,0,0]

colorset=[[0,160,50],[0,120,220],[150,30,150],[225,0,0],[255,100,0],[255,200,0]] #색 순서(g,b,p,r,o,y)
colornum2eng = ['g','b','p','r','o','y']

# resize
ver=1000 #행 크기 (세로)
hor=1000 #열 크기 (가로)
img_re=cv2.resize(img_cp,(ver,hor))

#흑백변환
img_gray=cv2.cvtColor(img_re,cv2.COLOR_BGR2GRAY)
#소벨 엣지 (def)
img_edge_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
img_edge_x = cv2.convertScaleAbs(img_edge_x)
img_edge_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
img_edge_y = cv2.convertScaleAbs(img_edge_y)
img_edge = cv2.addWeighted(img_edge_x, 1, img_edge_y, 1, 0)
#원검출 전처리
img_gray_cir = cv2.GaussianBlur(img_gray, (5,5), 0) #가우시안 블러
img_edge_cir = cv2.Laplacian(img_gray_cir, cv2.CV_8U, ksize=5) #라플라시안 엣지


for num in range(0,6):
    colordistance = [0,0,0,0,0,0] #색거리
    colorchk = [0,0,0] #측정한 rgb값

    M = cv2.getRotationMatrix2D((hor/2, ver/2), 60*num, 1.0)
    rotated = cv2.warpAffine(img_re, M, (hor, ver))
    rotated_def = cv2.warpAffine(img_edge, M, (hor, ver)) #불량검출 이미지 회전
    rotated_cir = cv2.warpAffine(img_edge_cir, M, (hor, ver)) #원검출 이미지 회전
    
    img_crop_def = rotated_def[269:310 , 425:575] #불량검출 이미지 크롭
    img_crop_cir = rotated_cir[:400 , 300:700] #원검출 이미지 크롭
    
    img_B = rotated[100:300 , 425:575,0] # B 이미지만 획득
    img_G = rotated[100:300 , 425:575,1] # G 이미지만 획득
    img_R = rotated[100:300 , 425:575,2] # R 이미지만 획득
    
    colorchk[0] = np.mean(img_R)
    colorchk[1] = np.mean(img_G)
    colorchk[2] = np.mean(img_B)

    
    for i in range(0,6):
        
        colordistance[i] = sqrt((colorset[i][0]-colorchk[0])**2+(colorset[i][1]-colorchk[1])**2+(colorset[i][2]-colorchk[2])**2)
        
        arr_color[num] = colordistance.index(min(colordistance)) #problem3 answer
        arr_color_eng[num] = colornum2eng[colordistance.index(min(colordistance))] #이해하기 쉽게 변환

    
    try:
        circles = cv2.HoughCircles(img_crop_cir, cv2.HOUGH_GRADIENT, 1, 200, param1 = 180, param2 = 50, minRadius = 80, maxRadius = 108)
        circles = np.uint16(np.around(circles))
    except:
        circles = 0 #원검출 안됬을경우 예외처리

    #불량 판별
    if np.mean(img_crop_def) > 75:
        arr_def[num] = 1
            
    #원 판별
    if np.mean(circles) != 0 :
        arr_cir[num] = 1
        
for num in range(0,6):
    if arr_def[num] == arr_cir[num]:
            arr_updn[num] = 1
            
circles = cv2.HoughCircles(img_edge_cir, cv2.HOUGH_GRADIENT, 1, 200, param1 = 180, param2 = 50, minRadius = 80, maxRadius = 108)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(img_re, (i[0], i[1]), i[2], (255, 255, 255), 5)
            
# redius = 300
# for i in range(0,6):
    # cv2.putText(img_re,str(arr_color_eng[i]),(int(460+redius*sin(90-i*60)),int(510+redius*cos(90-i*60))),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3, cv2.LINE_AA)

            
cv2.putText(img_re,str(arr_color_eng[0]),(470,380),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3, cv2.LINE_AA)
cv2.putText(img_re,str(arr_color_eng[1]),(610,420),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3, cv2.LINE_AA)
cv2.putText(img_re,str(arr_color_eng[2]),(610,600),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3, cv2.LINE_AA)
cv2.putText(img_re,str(arr_color_eng[3]),(470,680),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3, cv2.LINE_AA)
cv2.putText(img_re,str(arr_color_eng[4]),(340,600),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3, cv2.LINE_AA)
cv2.putText(img_re,str(arr_color_eng[5]),(340,420),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3, cv2.LINE_AA)


for num in range(0,6):
    arr_cir[num] = str(arr_cir[num])
    arr_updn[num] = str(arr_updn[num])
    arr_color[num] = str(arr_color[num])
    
cv2.putText(img_re,str(''.join(arr_color)),(780,90),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
cv2.putText(img_re,str(''.join(arr_updn)),(780,130),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
cv2.putText(img_re,str(''.join(arr_cir)),(780,160),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)



cv2.imshow('img1', img)
cv2.imshow('img2', img_re)
cv2.imshow('img3', img_gray)
cv2.imshow('img4', img_gray_cir)
cv2.imshow('img5', img_edge)
cv2.imshow('img6', img_edge_cir)
# cv2.imshow('imgb', img_B)
cv2.waitKey(0)
cv2.destroyAllWindows()
