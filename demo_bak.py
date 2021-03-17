import os
from PIL import Image

import sys
import pathlib

# 将 torchocr路径加到python路径里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))
sys.path.append(str(__dir__.parent))

import torch
from torch import nn
from torchvision import transforms
from torchocr.networks import build_model
from torchocr.datasets.det_modules import ResizeShortSize,ResizeFixedSize
from torchocr.postprocess import build_post_process

from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.utils import CTCLabelConverter

import cv2
from matplotlib import pyplot as plt
from torchocr.utils import draw_ocr_box_txt, draw_bbox
import argparse
import time
import numpy as np
import xlsxwriter

import pickle

# 构建一个类，作用类似于结构体
class cell:
	def __init__(self,lt,rd,belong):
		self.lt=lt
		self.rd=rd
		self.belong=belong   # 也是以角点来表示，一个单元格归属于它的左上角，如果归属同一个点，说明是同一个单元格！


class DetInfer:
	def __init__(self, model_path):
		ckpt = torch.load(model_path, map_location='cpu')
		cfg = ckpt['cfg']
		self.model = build_model(cfg['model'])
		state_dict = {}
		for k, v in ckpt['state_dict'].items():
			state_dict[k.replace('module.', '')] = v
		self.model.load_state_dict(state_dict)

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.model.to(self.device)
		self.model.eval()
		self.resize = ResizeFixedSize(736, False)
		self.post_proess = build_post_process(cfg['post_process'])
		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=cfg['dataset']['train']['dataset']['mean'], std=cfg['dataset']['train']['dataset']['std'])
		])

	def predict(self, img, is_output_polygon=False):
		# 预处理根据训练来
		data = {'img': img, 'shape': [img.shape[:2]], 'text_polys': []}
		data = self.resize(data)
		tensor = self.transform(data['img'])
		tensor = tensor.unsqueeze(dim=0)
		tensor = tensor.to(self.device)
		out = self.model(tensor)
		box_list, score_list = self.post_proess(out, data['shape'], is_output_polygon=is_output_polygon)
		box_list, score_list = box_list[0], score_list[0]
		if len(box_list) > 0:
			idx = [x.sum() > 0 for x in box_list]
			box_list = [box_list[i] for i, v in enumerate(idx) if v]
			score_list = [score_list[i] for i, v in enumerate(idx) if v]
		else:
			box_list, score_list = [], []
		return box_list, score_list


class RecInfer:
	def __init__(self, model_path):
		ckpt = torch.load(model_path, map_location='cpu')
		cfg = ckpt['cfg']
		self.model = build_model(cfg['model'])
		state_dict = {}
		for k, v in ckpt['state_dict'].items():
			state_dict[k.replace('module.', '')] = v
		self.model.load_state_dict(state_dict)

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.model.to(self.device)
		self.model.eval()

		self.process = RecDataProcess(cfg['dataset']['train']['dataset'])
		#self.converter = CTCLabelConverter(cfg['dataset']['alphabet'])
		self.converter = CTCLabelConverter('./torchocr/datasets/alphabets/ppocr_keys_v1.txt')

	def predict(self, img):
		# 预处理根据训练来
		img = self.process.resize_with_specific_height(img)
		# img = self.process.width_pad_img(img, 120)
		img = self.process.normalize_img(img)
		tensor = torch.from_numpy(img.transpose([2, 0, 1])).float()
		tensor = tensor.unsqueeze(dim=0)
		tensor = tensor.to(self.device)
		out = self.model(tensor)
		txt = self.converter.decode(out.softmax(dim=2).detach().cpu().numpy())
		return txt


class TabRecognition:
	def __init__(self,image):
		'''
		image: original image
		target: the combine of box and content
		'''
		self.image = image

	'''
	函数 islianjie（） 用于判断两点之间是否有连接。原理：两点之间 取多条 垂直的 横截线段，如果有像素值为0的，可以判断这两点是断开的。
	'''
	def islianjie(self,p1,p2,img): # 坐标p的格式是先y轴后x轴
		if p1[0]==p2[0]:   # y坐标相同，在同一横线
			for i in range(min(p1[1],p2[1]),max(p1[1],p2[1])+1):
				if sum( [ img[j,i] for j in range( max(p1[0]-5, 0), min(p1[0]+5, img.shape[0]) ) ] )==0: # img mask 格式也是先y后x
					return False
			return True

		elif p1[1]==p2[1]:  # x坐标相同，在同一竖线
			tmpsum = 0
			for i in range(min(p1[0],p2[0]), max(p1[0],p2[0])+1):   # y轴变化范围 
				if sum( [img[i,j] for j in range(max(p1[1]-5,0), min(p1[1]+5,img.shape[1])) ] ) == 0:   # x轴变化范围
					return False
			return True

		else:
			return False

	def crossingpointDetection(self):
		raw = self.image
		gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)    #  转换为灰度图片
		binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
		
		# 自适应获取核值
		rows, cols = binary.shape
		scale = 20
		scale2 = 15
		# 形态学处理，识别横线:
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
		kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale2, 1))
		eroded = cv2.erode(binary, kernel, iterations=1)
		dilated_col = cv2.dilate(eroded, kernel1, iterations=1)
		#cv2.imwrite(respath+"1_横向形态学.jpg", dilated_col)

		# 形态学处理，识别竖线：
		# scale = 40#scale越大，越能检测出不存在的线
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
		kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale2))
		eroded = cv2.erode(binary, kernel, iterations=1)
		dilated_row = cv2.dilate(eroded, kernel2, iterations=1)
		#cv2.imwrite(respath+"2_竖向形态学.jpg", dilated_row)

		# 将识别出来的横竖线合起来
		bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)  # 对二值图进行 与操作，即可求得交点
		#cv2.imwrite(respath+"3_横向竖向交点.jpg", bitwise_and)

		# 标识表格轮廓
		merge = cv2.add(dilated_col, dilated_row)
		ret,binary = cv2.threshold(merge, 127, 255, cv2.THRESH_BINARY)
		self.merge = merge.copy()
		#cv2.imwrite(respath+"4_横竖交点阈值化.jpg", binary)

		ys, xs = np.where(bitwise_and > 0)

		'''
		关键点： 利用相邻位置信息，过滤重复直线。输出为： 交点的横纵坐标数组y_point_arr， x_point_arr
		'''
		# 交点的横纵坐标数组
		y_point_arr = []
		x_point_arr = []
		# 通过排序，排除掉相近的像素点，只取相近值的最后一点
		# 这个3就是两个像素点的距离，不是固定的，根据不同的图片会有调整，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
		i = 0
		sort_x_point = np.sort(xs)
		for i in range(len(sort_x_point) - 1):
			if sort_x_point[i + 1] - sort_x_point[i] > 3:
				x_point_arr.append(sort_x_point[i])
			i = i + 1
		x_point_arr.append(sort_x_point[i])  # 要将最后一个点加入

		i = 0
		sort_y_point = np.sort(ys)
		for i in range(len(sort_y_point) - 1):
			if sort_y_point[i + 1] - sort_y_point[i] > 3:
				y_point_arr.append(sort_y_point[i])
			i = i + 1
		y_point_arr.append(sort_y_point[i])
		# print("sorted coor:")
		# print(len(sort_x_point),len(sort_y_point))
		# print("filtered coor:")
		# print(len(x_point_arr),len(y_point_arr))

		self.y_crossingpoint_arr = y_point_arr
		self.x_crossingpoint_arr = x_point_arr
		return y_point_arr, x_point_arr

	
	def cellRecognition(self):
		y_point_arr = self.y_crossingpoint_arr
		x_point_arr = self.x_crossingpoint_arr 
		
		# 计算所有可能的长和宽
		h_list = [y_point_arr[i+1]-y_point_arr[i] for i in range(len(y_point_arr)-1)]
		w_list = [x_point_arr[i+1]-x_point_arr[i] for i in range(len(x_point_arr)-1)]

		lt_list_x = x_point_arr[:-1]  # 取前面的n-1个值，最后一个不取
		lt_list_y = y_point_arr[:-1]
		rd_list_x = x_point_arr[1:]   # 从第2个值开始，第一个值不取，共n-1个值
		rd_list_y = y_point_arr[1:]

		d={}
		for i in range(len(lt_list_x)):
			for j in range(len(lt_list_y)):
				d['cell_{}_{}'.format(i,j)] = cell( [lt_list_x[i],lt_list_y[j]], [rd_list_x[i],rd_list_y[j]], [lt_list_x[i],lt_list_y[j]])

		for i in range(len(lt_list_x)):
			for j in range(len(lt_list_y)):
				## p点格式为(y,x)。假设 左上角 lt(y1,x1), 右下角 rd(y2,x2) ==> 左下角 p1(y2,x1), 右上角 p3(y1,x2)
				p1 = [d['cell_{}_{}'.format(i,j)].rd[1], d['cell_{}_{}'.format(i,j)].lt[0]]  #左下点 
				p2 = [d['cell_{}_{}'.format(i,j)].rd[1], d['cell_{}_{}'.format(i,j)].rd[0]]  #右下点 
				p3 = [d['cell_{}_{}'.format(i,j)].lt[1], d['cell_{}_{}'.format(i,j)].rd[0]]  #右上点
				## 查看两点之间是否连接，确定单元格归属
				if not self.islianjie(p1,p2,self.merge):
					d['cell_{}_{}'.format(i,j+1)].belong = d['cell_{}_{}'.format(i,j)].belong
				if not self.islianjie(p2,p3,self.merge):
					d['cell_{}_{}'.format(i+1,j)].belong=d['cell_{}_{}'.format(i,j)].belong

		crop_list={}
		for i in range(len(lt_list_x)):
			for j in range(len(lt_list_y)):
				## crop_list字典以 “归属值” 为key，然后遍历所有单元格（一定要按顺序！），可以合并单元格
				crop_list['{},{}'.format(d['cell_{}_{}'.format(i,j)].belong[0], d['cell_{}_{}'.format(i,j)].belong[1])]= d['cell_{}_{}'.format(i,j)].rd
		
		self.crop_list = crop_list
		return crop_list,h_list, w_list
	
	def detnrec(self):
		self.crossingpointDetection()
		rop_list, h_list, w_list = self.cellRecognition()
		#return crop_list
		return self.crop_list, h_list, w_list


def generateExcelFile(path,filename,bboxes_loc,rec_content,crop_list,h_list,w_list):
	workbook = xlsxwriter.Workbook(os.path.join(path + '/' + filename))     # 创建新的工作簿
	worksheet = workbook.add_worksheet()   # 添加新的工作表
	# 先按行列数设置单元格，不管单元格合并格式
	col_alpha=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

	for i in range(len(w_list)):
		worksheet.set_column('{}:{}'.format(col_alpha[i],col_alpha[i]),w_list[i]/6) 
	for j in range(len(h_list)):
		worksheet.set_row(j+1,h_list[j])
		
	merge_format = workbook.add_format({
		'bold':     True,
		'border':   1,
		'align':    'left',#水平居中
		'valign':   'vcenter',#垂直居中
		#'fg_color': '#D7E4BC',#颜色填充
	})

	header_format = workbook.add_format({
		'bold':     True,
		'border':   1,
		'align':    'center',#水平居中
		'valign':   'vcenter',#垂直居中
		#'fg_color': 'blue',#颜色填充
	})

	def is_inside(cell, box):
		c1,c2,c3,c4 = cell
		b1,b2,b3,b4 = box
		if b1>c1 and b2>c2 and b3<c3 and b4<c4:
			return True
		else:
			return False
	
	tmpmax=0
	tmpmin=1e6
	zlt=[]  # 整张表格的最左上角点坐标
	zrd=[]  # 整张表格的最右下角点坐标
	for key in crop_list.keys():
		lt=[int(i) for i in key.split(',')]
		rd=crop_list[key]
		#cv2.imwrite('/home/elimen/Data/dbnet_pytorch/test_results_cell/{}.jpg'.format(key),raw[lt[1]:rd[1],lt[0]:rd[0]])  # 图片裁剪格式 img[y1:y2,x1:x2] or img[(y1,x1),(y2,x2)]

		if sum(rd)>tmpmax:
			zrd=rd
			tmpmax=sum(rd)
		if sum(lt)<tmpmin:
			zlt=lt
			tmpmin=sum(lt)
		
	'''
	collect and write the header first
	'''
	stored_index = []
	for key in crop_list.keys():
		lt = [int(i) for i in key.split(',')]
		rd = crop_list[key]

		for i in range(len(bboxes_loc)):
			box = bboxes_loc[i]
			cell = [lt[0],lt[1],rd[0],rd[1]]
			if is_inside(cell,box):
				stored_index.append(i)
			
	tmp_index = [ind for ind in range(len(bboxes_loc))]
	header_index = []
	for j in range(len(tmp_index)):
		if tmp_index[j] not in stored_index:
			header_index.append(j)
	if header_index:
		header = ''
		for j in range(len(header_index)):
			header += rec_content[j]+'\n'
	worksheet.set_row(0, sum(h_list)/len(h_list)*4)
	worksheet.merge_range('{}{}:{}{}'.format('A',1,chr(ord('A')+len(w_list)-1),1),'{}'.format(header),header_format)  # 合并单元格

	'''
	根据crop_list, 遍历每个单元格，然后分配行列序号
	'''
	for key in crop_list.keys():
		lt = [int(i) for i in key.split(',')]
		rd = crop_list[key]

		content = []
		for i in range(len(bboxes_loc)):
			box = bboxes_loc[i]
			cell = [lt[0],lt[1],rd[0],rd[1]]
			if is_inside(cell,box):
				content.append(rec_content[i].split('\n')[0])
	
		lt_dist2ori = [lt[0]-zlt[0],lt[1]-zlt[1]]
		rd_dist2ori = [rd[0]-zlt[0],rd[1]-zlt[1]]

		## 水平方向 
		for i in range(len(w_list)+1):
			# 左上角
			if lt_dist2ori[0]==sum(w_list[:i]):
				lt_col=chr(ord('A')+i)
				#print(lt_col)
			# 右下角
			if rd_dist2ori[0]==sum(w_list[:i]):
				rd_col=chr(ord('A')+i-1)
				#print(rd_col)
		## 竖直方向
		for i in range(len(h_list)+1):
			# 左上角
			if lt_dist2ori[1]==sum(h_list[:i]):
				lt_row=i+2
				#print(lt_row)
			# 右下角
			if rd_dist2ori[1]==sum(h_list[:i]):
				rd_row=i+1
				#print(rd_row)

		contents = ''
		if content:
			for k in range(len(content)-1):
				contents += content[k] + '\n'
			contents += content[len(content)-1]

		if lt_col==rd_col and lt_row==rd_row:
			worksheet.write('{}{}'.format(lt_col,lt_row),'{}'.format(contents),merge_format)   # 写入内容
		else:
			worksheet.merge_range('{}{}:{}{}'.format(lt_col,lt_row,rd_col,rd_row),'{}'.format(contents),merge_format)  # 合并单元格
	
	workbook.close()
	return workbook

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='DocOCR infer')
	parser.add_argument('--img_path', type=str, help='img path for predict',default='./test_images/mt02.png')
	parser.add_argument('--res_path', type=str, help='res path for predict',default='./test_results/')
	args = parser.parse_args()
	
	# modeldet = DetInfer(args.modeldet_path)
	# modelrec = RecInfer(args.modelrec_path)
	# with open('modeldet.pkl', 'wb') as det:
	# 	pickle.dump(modeldet, det)
	# with open('modelrec.pkl', 'wb') as rec:
	# 	pickle.dump(modelrec, rec)
	# print("Model saved.")

	with open('./modeldet.pkl', 'rb') as det:
		modeldet = pickle.load(det)
	with open('./modelrec.pkl', 'rb') as rec:
		modelrec = pickle.load(rec)
	print("Model loaded.")

	img_name = args.img_path.split('/')[-1].split('.')[0]
	res_path = args.res_path
	img_name = args.img_path.split('/')[-1].split('.')[0]
	imageres_name = img_name + '_detBoxes.jpg'
	fileres_name = img_name + '.xlsx'
	

	img = cv2.imread(args.img_path)
	img_bak = img.copy()
	
	box_list, score_list = modeldet.predict(img, is_output_polygon=False)
	
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = draw_bbox(img, box_list)
	cv2.imwrite(res_path+imageres_name,img)

	txt_file = os.path.join(res_path, img_name + '_recContents.txt')
	txt_f = open(txt_file, 'w')


	'''
	output the bbox corner and text recognition result
	'''
	imgcroplist = []
	bbox_cornerlist = []
	for i, box in enumerate(box_list):
		pt0,pt1,pt2,pt3=box

		imgout = img_bak[int(min(pt0[1],pt1[1]))-4 :int(max(pt2[1],pt3[1])) +4,int(min(pt0[0],pt3[0]))-4:int(max(pt1[0],pt2[0]))+4]
		imgcroplist.append(imgout)
		#cv2.imwrite(res_path+imageres_name.split('.')[0]+'_'+str(i)+'.jpg',imgout)

		box_corner = [int(pt0[0]),int(pt0[1]),int(pt2[0]),int(pt2[1])]
		bbox_cornerlist.append(box_corner)
	bbox_cornerlist.reverse()
		
	rec_cont = [] 
	for i in range(len(imgcroplist)-1,-1,-1):
		out = modelrec.predict(imgcroplist[i])
		rec_cont.append(out[0][0]) 

		txt_f.write(str(bbox_cornerlist[i]))
		txt_f.write(out[0][0]+ '\n')	
	txt_f.close()

	tab_rec = TabRecognition(img_bak)
	crop_list,height_list, width_list= tab_rec.detnrec()
	generateExcelFile(res_path,fileres_name,bbox_cornerlist,rec_cont,crop_list,height_list,width_list)

	print("Mission complete.")


