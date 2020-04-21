import cv2, numpy as np
import utils

########
path = "1.jpg"
imgWidth = 300
imgHeight = 400
questions = 5
choices = 5
ans = [1,2,0,1,4]
webCamFeed = True
########

cap = cv2.VideoCapture(0)
cap.set(10,150)

while True:
	if webCamFeed:
		success,pic = cap.read()
	else:
		pic = cv2.imread(path)

	pic = cv2.resize(pic,(imgWidth, imgHeight))

	picContours = pic.copy()
	picBiggestContours = pic.copy()
	picFinal = pic.copy()


	picGray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
	picBlur = cv2.GaussianBlur(picGray , (5,5), 1)
	picCanny = cv2.Canny(picBlur, 10, 50)


	try:
		## finding all contours
		contours, heirarchy = cv2.findContours(picCanny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		cv2.drawContours(picContours, contours, -1, (0,255,0), 1)

		# find rectangles
		rectCon = utils.rectContour(contours)
		biggestContour = utils.getCornerPoints(rectCon[0]) 				#get four co-ord of biggest rect
		gradePoints = utils.getCornerPoints(rectCon[1])

		if biggestContour.size != 0 and gradePoints.size != 0:
			cv2.drawContours(picBiggestContours, biggestContour, -1, (255,0,0), 7)
			cv2.drawContours(picBiggestContours, gradePoints, -1, (0,0,255), 7)
			#print("big cont:",biggestContour.shape)	--> (4,1,2)
			biggestContour = utils.reorderPoints(biggestContour)
			gradePoints    = utils.reorderPoints(gradePoints)

			# get bird eye view of biggest contour
			pt1 = np.float32(biggestContour)
			pt2 = np.float32([ [0,0],[imgWidth,0],[0,imgHeight],[imgWidth, imgHeight] ])
			matrix = cv2.getPerspectiveTransform(pt1, pt2)
			imgWarpColored = cv2.warpPerspective(pic , matrix , (imgWidth, imgHeight))

			#get bird view of second biggest
			ptG1 = np.float32(gradePoints)
			ptG2 = np.float32([ [0,0],[325,0],[0,150],[325, 150] ])
			matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
			imgWarpGrade = cv2.warpPerspective(pic , matrixG , (325, 150))
			#cv2.imshow("grade",imgWarpGrade)

			#apply threshold in biggest contour to get the grades marked
			imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
			imgThresh   = cv2.threshold(imgWarpGray, 170,255,cv2.THRESH_BINARY_INV)[1]

			#get each bubble as separate image re
			boxes = utils.splitBoxes(imgThresh)
			#print(cv2.countNonZero(boxes[1]) , cv2.countNonZero(boxes[2]))
			#cv2.imshow("test",boxes[2])

			#GET NON ZERO PIXEL VALUE OF EACH BOX
			myPixelVal = np.zeros((questions, choices))
			countC = 0
			countR = 0

			for image in boxes:
				totalPixels = cv2.countNonZero(image)
				myPixelVal[countR][countC] = totalPixels
				countC += 1
				if (countC == choices):	
					countR += 1
					countC = 0

			myIndex = []
			for x in range(questions):
				arr = myPixelVal[x]
				myIndexVal = np.where(arr == np.amax(arr))
				myIndex.append(myIndexVal[0][0])
			#ans marked -->
			#print(myIndex)

			#GRADE THE ANSWERS
			grading = []
			for x in range(questions):
				if ans[x] == myIndex[x]:
					grading.append(1)
				else:	grading.append(0)

			#	SCORING CALCULATION
			score = (sum(grading)/questions) *100
			print(score)

			#	DISPLAYING ANSWERS
			imgResult = imgWarpColored.copy()
			imgResult = utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
			imRawDrawing = np.zeros_like(imgWarpColored)
			imRawDrawing = utils.showAnswers(imRawDrawing, myIndex, grading, ans, questions, choices)

			#	INVERSE PERS --> BIRD VIEW SE ORIGNINAL IMAGE TAK...MEANS POINTS LE GREEN AND RED AND MARK ON MAIN SCREEN
			invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
			imgInverseWarp = cv2.warpPerspective(imRawDrawing , invMatrix , (imgWidth, imgHeight))

			#	GET GRADE IN THE BOX
			imgRawGrade = np.zeros_like(imgWarpGrade)
			cv2.putText(imgRawGrade, str(int(score)) + "%", (60, 100),cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255),3)
			#cv2.imshow("grade",imgRawGrade)

			#	INVERSE GRADE PERSPECTIVE 
			invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
			imgInvGrade = cv2.warpPerspective(imgRawGrade , invMatrixG , (imgWidth, imgHeight))


			#	FINAL IMAGE
			picFinal = cv2.addWeighted(picFinal, 1, imgInverseWarp, 1, 0)
			picFinal = cv2.addWeighted(picFinal, 1, imgInvGrade, 1, 0)

		

		imgBlank = np.zeros_like(pic)
		imgArray = ([pic,picGray,picBlur,picCanny],[picContours, picBiggestContours, imgWarpColored, imgThresh],[imgResult, imRawDrawing, imgInverseWarp, picFinal])

	except:
		imgBlank = np.zeros_like(pic)
		imgArray = ([pic,picGray,picBlur,picCanny],[imgBlank, imgBlank, imgBlank, imgBlank],[imgBlank, imgBlank, imgBlank, imgBlank])		
	
	lables = [["Original","Gray","Edges","Canny"],
	              ["Contour","biggest contour","Warpped","Threshold"],
	              ["Result","Raw drawing","Inv Warp","Final"]]

	imgStack = utils.stackImages(imgArray, 0.5, lables)
	cv2.imshow("pakode",imgStack)
	

	#	SAVE THE IMAGE
	if cv2.waitKey(1) and 0xFF == ord('s'):
		cv2.imwrite('FinalResult.jpg', picFinal)
		cv2.waitKey(300)

	if cv2.waitKey(1) & 0xFF == ord('q'):
        break
