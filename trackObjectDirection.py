#OPEN CV 2.*
import cv2
import numpy as np 
import math


def findBig(hull):
	maxDAP = [] 
	cpointX = 0;cpointY = 0;pointX  = 0;pointY  = 0;maxdis  = 0
	maxXA   = 0;maxYA   = 0;maxXB   = 0;maxYB   = 0

	for i,points in enumerate(hull):
		cpointX += points[0][0]
		cpointY += points[0][1]

	for i,points in enumerate(hull):
		for newpoint in hull[i+1:]:
			dis = math.sqrt( (points[0][0]-newpoint[0][0])**2 + (points[0][1]-newpoint[0][1])**2 )
			if dis>maxdis:
				maxdis = dis
				maxXA   =  points[0][0]
				maxYA   =  points[0][1]
				maxXB   =  newpoint[0][0]
				maxYB   =  newpoint[0][1]

	returnList = []
	returnList.append((int(cpointX/len(hull)),int(cpointY/len(hull))))
	returnList.append((maxXA,maxYA))
	returnList.append((maxXB,maxYB))
	returnList.append(maxdis)

	return returnList


def checkMove(lines,points):
	if( ((lines[2][0]<points[0] and lines[3][0]>points[0]) and (lines[2][0]<points[1] and lines[3][0]>points[1])) or ((lines[0][0]<points[0] and lines[3][0]>points[0]) and (lines[2][0]<points[0] and lines[1][0]>points[1])) )  :
		return "Moving Back from Camera"

	elif ((lines[0][0]>points[0]) or (lines[0][0]>points[1])) and ((lines[1][0]<points[0]) or (lines[1][0]<points[1])):
		return "Moving Toward Camera"

	elif( (lines[0][0]>points[0]) or (lines[0][0]>points[1])):
		return "Moving Right"

	elif ( ((lines[1][0]<points[0]) or (lines[1][0]<points[1]))):
		return "Moving Left"

	else:
		return "None"


if __name__ == '__main__':

	lines = [[(180,0),(180,500)],[(620,0),(620,500)],[(250,0),(250,500)],[(550,0),(550,500)]]

	co = 0

	cap = cv2.VideoCapture(0)

	while True:
		try:

			_,frame = cap.read()

			frame = cv2.resize(frame, (800,500))

			grayImage = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			hsvImage  = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

			lower = np.array([5,130,130],dtype='uint8')
			upper = np.array([15,255,255],dtype='uint8')

			binaryImage =  cv2.inRange(hsvImage, lower, upper)

			ksiz = 5
			kernel = np.ones((ksiz,ksiz),dtype='uint8')

			binaryImage = cv2.dilate(binaryImage,kernel,iterations=2)

			contour,_ = cv2.findContours(binaryImage.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

			maxDis = 0;maxXA   = 0;maxYA   = 0;maxXB   = 0;maxYB   = 0;cx = 0;cy = 0

			contourImage = np.ones((frame.shape),dtype='uint8')

			cv2.line(contourImage, lines[0][0], lines[0][1], (255,0,0), 1)
			cv2.line(contourImage, lines[1][0], lines[1][1], (255,0,0), 1)
			cv2.line(contourImage, lines[2][0], lines[2][1], (255,0,0), 1)
			cv2.line(contourImage, lines[3][0], lines[3][1], (255,0,0), 1)

			for c in contour:

				accuracy = 0.0001*cv2.arcLength(c, True)
				approx   = cv2.approxPolyDP(c, accuracy, True)
				hull     = cv2.convexHull(approx)
				# cv2.drawContours(contourImage, [hull], 0, (255,255,255), 1)
				rList  = findBig(hull) 
				if rList[3]>maxDis:
					maxDis = rList[3]
					# cx,cy = rList[0][0],rList[0][1]
					maxXA,maxYA = rList[1][0],rList[1][1]
					maxXB,maxYB = rList[2][0],rList[2][1]		


			d = math.sqrt(((maxXA- maxXB)**2)+((maxYA- maxYB)**2)) 

			if d>20:

				contourImage = cv2.bitwise_and(frame, frame, mask=binaryImage) # DO THIS IF YOU WANT TO HIDE YOUR FACE  AND BACKGROUND 

				cv2.line(contourImage, lines[0][0], lines[0][1], (255,0,0), 1)
				cv2.line(contourImage, lines[1][0], lines[1][1], (255,0,0), 1)
				cv2.line(contourImage, lines[2][0], lines[2][1], (255,0,0), 1)
				cv2.line(contourImage, lines[3][0], lines[3][1], (255,0,0), 1)

				#contourImage = frame.copy() # OTHERWISE DO THIS
				cx = int((maxXA+maxXB)/2)		
				cy = int((maxYA+maxYB)/2)		
						
				cv2.circle(contourImage, (cx,cy), 5, (0,0,255), -1)
				cv2.circle(contourImage, (maxXA,maxYA), 5, (0,0,255), -1)
				cv2.circle(contourImage, (maxXB,maxYB), 5, (0,0,255), -1)
				cv2.rectangle(contourImage, (maxXA,maxYA), (maxXB,maxYB), (0,0,255), 2)

				points = []
				points.append((maxXA,maxYA))
				points.append((maxXB,maxYB))
				direction = checkMove(lines, points)

				cv2.putText(contourImage, direction, (400,400), 2, 0.5, (0,255,0) , 1)


			cv2.imshow('Image', contourImage)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		except:
			cv2.destroyAllWindows()
			raise 

	cv2.destroyAllWindows()