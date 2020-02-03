import cv2
import numpy as np
import os
from matplotlib import pyplot as plt 

class sol:
	def __init__(self):
		self.root_dir = os.getcwd()
		self.pic_dir = os.path.join(self.root_dir,'source')
		self.Intrinsic = np.array([[2225.49585482, 0, 1025.5459589], [0, 2225.18414074, 1038.58578846], [0, 0, 1]])
		self.distortion = np.array([[-0.12874225, 0.09057782, -0.00099125, 0.00000278, 0.0022925]])
		self.bmp1 = np.array(
			[[-0.97157425, -0.01827487, 0.23602862, 6.81253889], [0.07148055, -0.97312723, 0.2188925, 3.37330384],
			 [0.22568565, 0.22954177, 0.94677165, 16.74572319]])
		self.bmp2 = np.array(
			[[-0.8884799,-0.14530922,-0.4353030,3.3925504], [0.07148066,-0.98078915,0.18150248,4.36149229],
			 [-0.45331444,0.13014556,0.88179825,22.15957429]])
		self.bmp3 = np.array(
			[[-0.52390938,0.22312793,0.82202974,2.68774801], [0.00530458,-0.96420621,0.26510049,4.70990021],
			 [0.85175749,0.14324914,0.50397308,12.98147662]])
		self.bmp4 = np.array(
			[[-0.63108673,0.53013053,0.566296,1.22781875], [0.13263301,-0.64553994,0.75212145,3.48023006],
			 [0.76428923,0.54976341,0.33707888,10.9840538]])
		self.bmp5 = np.array(
			[[-0.87676843,-0.23020567,0.42223508,4.43641198], [0.19708207,-0.97286949,-0.12117596,0.67177428],
			 [0.43867502,-0.02302829,0.89835067,16.24069227]])
		self.point = np.array([[1, 1, 0], [1, 5, 0], [5, 5, 0], [5, 1, 0],[3, 3, -4]],dtype=np.float32)
		self.dict = {1: '1.bmp', 2: '2.bmp', 3: '3.bmp', 4: '4.bmp', 5: '5.bmp'}
		self.dict_extrinsic = {1: self.bmp1,2: self.bmp2, 3: self.bmp3, 4: self.bmp4, 5: self.bmp5}
		self.AR_result = []
	def q1(self):
		imgL = cv2.imread(os.path.join(self.pic_dir,'imL.png'),0)
		imgR = cv2.imread(os.path.join(self.pic_dir,'imR.png'),0)
		height,width = imgL.shape
		imgL = cv2.equalizeHist(imgL)
		imgR = cv2.equalizeHist(imgR)
		imgL = cv2.GaussianBlur(imgL,(5,5),0)
		imgR = cv2.GaussianBlur(imgR,(5,5),0)
		#stereo =cv2.StereoBM_create(64,9)
		# print(stereo.getNumDisparities())
		# stereo.setMinDisparity(1)
		# stereo.setPreFilterType(0)
		# stereo.setPreFilterSize(3*3)
		# stereo.setPreFilterCap(63)
		# stereo.setUniquenessRatio(10)
		# stereo.setDisp12MaxDiff(1)
		# stereo.setSpeckleWindowSize(49)
		# stereo.setSpeckleRange(100)
		stereo = cv2.StereoSGBM_create(
			minDisparity=1,
			numDisparities=64,
			blockSize=3*3,
			P1=8 * 3 * 7 ** 2,
			P2=32 * 3 * 7 ** 2,
			disp12MaxDiff=1,
			uniquenessRatio=10,
			speckleWindowSize=100,
			speckleRange=32
		)
		disparity = stereo.compute(imgL,imgR)
		disparity = (disparity-0)/16
		Dmax = disparity.max()
		Dmin = disparity.min()
		for i in range(0,height-1):
			for j in range(0,width-1):
				disparity[i][j] = round((255-0)*(disparity[i][j]-Dmin)/(Dmax-Dmin)+0)
		plt.imshow(disparity,'gray')
		plt.show()
	def q2(self):
		bgSub = os.path.join(self.pic_dir,'bgSub.mp4')
		ori = cv2.VideoCapture(bgSub)
		model = cv2.createBackgroundSubtractorKNN(history=50,detectShadows=True)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		cv2.namedWindow('original')
		cv2.namedWindow('result')
		if ori.isOpened():
			while True:
				ret,prev = ori.read()
				if ret == True :
					tmp   = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
					result = model.apply(tmp)
					_,result = cv2.threshold(result, 15, 255, cv2.THRESH_BINARY)
					result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
					cv2.imshow('original',prev)
					cv2.imshow('result',result)
				else :
					break
				if cv2.waitKey(30) == 27 :
					break
		cv2.destroyAllWindows()
	def Proprocessing(self):
		file = os.path.join(self.pic_dir,'featureTracking.mp4')
		video = cv2.VideoCapture(file)
		ret, pic = video.read()
		keypoint,pic = self.dect(pic,0)
		cv2.namedWindow('result')
		cv2.imshow('result', pic)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	def tracking(self):
		lk_params = dict(winSize=(21, 21),
						 maxLevel=0,
						 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,1000,0),
						 flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
		file = os.path.join(self.pic_dir, 'featureTracking.mp4')
		video = cv2.VideoCapture(file)
		ret, old_frame = video.read()
		old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
		p0,img= self.dect(old_frame,0)
		mask = np.zeros_like(old_frame)
		idx = 1
		if video.isOpened():
			while True:
				ret,frame = video.read()
				if ret == True:
					tmp = frame.copy()
					frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
					p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
					good_new = p1[st == 1]
					good_old = p0[st == 1]
					for i, (new, old) in enumerate(zip(good_new, good_old)):
						a, b = new.ravel()
						c, d = old.ravel()
						mask = cv2.line(mask, (a, b), (c, d), (0,0,255), 2)
						sz = np.int(11 / 2)
						frame = cv2.rectangle(frame, (np.int(a - sz), np.int(b - sz)), (np.int(a + sz),np.int( b + sz)), color=(0,0,255), thickness=-1)
					result= cv2.add(frame, mask)
					cv2.imshow('frame', result)
					old_gray = frame_gray.copy()
					p0 = good_new.reshape(-1, 1, 2)
					idx += 1
				else :
					break
				if cv2.waitKey(10) == 27 :
					break
			cv2.destroyAllWindows()


	def dect(self,img,sel=0):
		if sel == 1:
			tmp =img
		else :
			tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		params = cv2.SimpleBlobDetector_Params()
		params.filterByCircularity=True
		params.minCircularity=  0.83
		params.maxCircularity = 1
		params.filterByArea=True
		params.minArea=30
		params.maxArea=100
		params.filterByConvexity=True
		params.minConvexity=0.95
		params.maxConvexity=1
		params.filterByInertia=True
		params.minInertiaRatio = 0.49
		params.maxInertiaRatio = 1
		params.minThreshold = 90;
		params.maxThreshold = 170;
		detector = cv2.SimpleBlobDetector_create(params)
		keypoints = detector.detect(tmp)
		track = np.zeros(shape=(7,1,2),dtype=np.float32)
		for i in range(0, len(keypoints)):
			x, y = np.int(keypoints[i].pt[0]), np.int(keypoints[i].pt[1])
			track[i,0,0] = x
			track[i,0,1] = y
			sz = np.int(11 / 2)
			img = cv2.rectangle(img, (x - sz, y - sz), (x + sz, y + sz), color=(0,0,255), thickness=2)
		return  track,img

	def q4(self):
		self.img = []
		for i in range(1,6):
			img = self.drawpyra(i)
			self.AR_result.append(img)
	def showq4(self):
		if len(self.AR_result) == 0 :
			self.q4()
		cv2.namedWindow("result",cv2.WINDOW_NORMAL)
		for i in range(0,5):
			cv2.imshow("result",self.AR_result[i])
			if cv2.waitKey(500) == 27 :
					break
		cv2.destroyAllWindows()

	def drawpyra(self, index):
		file = os.path.join(self.pic_dir,self.dict[index])
		trans = self.dict_extrinsic[index][:,3:]
		rot = self.dict_extrinsic[index][:,0:3]
		img = cv2.imread(file)
		point2d, _ = cv2.projectPoints(self.point, rot, trans, self.Intrinsic, self.distortion)
		for i in range(0,len(point2d)-1):
			if i == 3 :
				y = 0
			else :
				y = i+1
			cv2.line(img,tuple(point2d[i][0]),tuple(point2d[y][0]),(0,0,255),10)
			cv2.line(img, tuple(point2d[i][0]), tuple(point2d[4][0]),(0, 0, 255),10)
		return  img


