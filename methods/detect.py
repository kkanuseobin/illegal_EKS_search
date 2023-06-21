### 라이브러리 import
from ultralytics import YOLO   # YOLO 라이브러리 import
import itertools, math, cv2, os
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # 백엔드 지정, yolov8에선 필요한 듯
import matplotlib.pyplot as plt

### yolo model 적용 (person, scooter, helmet, road)
person_model = YOLO('./make_models/yolov8n.pt')   # persone (detect)
scooter_model = YOLO('./make_models/scooter.pt') # scooter (detect)
helmet_model = YOLO('./make_models/helmet.pt')   # helmet (detect)
road_model = YOLO('./make_models/road.pt')       # road (segment)

### Main function (detection + segment => results)
def search(video, folder_name, main_idx):
	# 사람과 스쿠터의 박스와 이미지 가져오기
	p_boxes, p_imgs = find_box(person_model, video, goal='person')
	s_boxes, s_imgs = find_box(scooter_model, video, goal=None)

	output = []  # 결과 저장 list
	matching = []

	'''
	person과 scooter 매칭 정보 1차 확인
	'''
	p_xyxy, s_xyxy = [], []
	for idx, p_box in enumerate(p_boxes):
		if len(p_box.xyxy) != 0:
		    try:
		    	p_xyxy.append((idx, p_box.id.tolist(), p_box.xyxy.tolist(), p_box.xywh.tolist()))
		    except:pass
		        

	for idx, s_box in enumerate(s_boxes):
		try:
			if s_box is not None:
				s_xyxy.append((idx, s_box.id.tolist(), s_box.xyxy.tolist(), s_box.xywh.tolist()))
		except: pass

	for idx, pid, p, p_xywh in (p_xyxy):
		for jdx, sid, s, s_xywh in (s_xyxy):
		    if idx == jdx:
		        for i, p_i in enumerate (p):
		            for j, s_i in enumerate (s):
		                iou = IoU(p_i[0], p_i[1], p_i[2], p_i[3], s_i[0], s_i[1], s_i[2], s_i[3])
		                if iou > 0.2:
		                    matching.append((idx, pid[i], sid[j], i, j))

	count_dict = {}
	for item in matching:
		key = item[1:3]
		count_dict[key] = count_dict.get(key, 0) + 1

	riding = []
	ride = None
	for key, count in count_dict.items():
		if count >= 30:
		    min_idx = None
		    max_idx = None
		    c_riding = 0
		    
		    for idx, pid, sid, i, j in matching:
		        if (pid, sid) == key:
		            if min_idx is None or idx < min_idx: min_idx = idx
		            if max_idx is None or idx > max_idx: max_idx = idx
		    
		    
		            # 첫 번째 인덱스와 마지막 인덱스에 대한 박스 좌표 가져옴
		            f_px, f_py, _, __ = p_boxes[min_idx].xywh.tolist()[i]
		            l_px, l_py, _, __ = p_boxes[max_idx].xywh.tolist()[i]

		            f_sx, f_sy, _, __ = s_boxes[min_idx].xywh.tolist()[j]
		            l_sx, l_sy, _, __ = s_boxes[max_idx].xywh.tolist()[j]

		            # 코사인 유사도 계산
		            similarity = cosine_similarity(f_px, f_py, l_px, l_py, f_sx, f_sy, l_sx, l_sy)
		            fd, ld = euclidean_distance(f_px, f_py, l_px, l_py, f_sx, f_sy, l_sx, l_sy)
		            if (similarity > 0.5) and ((fd+ld)/2 < 200):  # 0은 세팅된 임계값
		                c_riding += 1
		#                 riding.append((key, min_idx, max_idx))
		                if c_riding > 30:
		                    ride = (key, (min_idx, max_idx), i, j, similarity, (fd, ld))

		    riding.append(ride)
		    print(riding)
		    output.append((key[0],0,0,0))

	'''
	다인 탑승 여부 확인
	'''
	output = list(set(output))
	print("최초세팅: ", output)
	try:
		result = list(set([tpl[0] for tpl in riding if tpl[1] == riding[0][1]]))


		for i, (p_id, a, b, c) in enumerate (output):
			if len(output) == 1:
				break
			for (pr_id, _) in result:
				if p_id == pr_id:
				    updated_tuple = (p_id, 1, b, c)
				    output[i] = updated_tuple
		print("다인탑승: ", output)
	except: pass

	'''
	헬멧 착용 여부 확인
	'''

	for (pid, sid), (min_idx, max_idx), ri, rj, _, (_,_) in riding:
		index = 0
		c, area = 0, 0 # 변수 초기화, c는 겹치는 부분이고 탐색 픽셀 수
		for idx, m_pid, m_sid, mi, mj in matching:
			print(pid, sid, ri, rj, m_pid, m_sid, mi, mj)
			if (pid, sid, ri, rj) == (m_pid, m_sid, mi, mj):
				p_img = p_imgs[idx]
				seg_img = p_img.copy()
				px1, py1, px2, py2 = p_boxes[idx].xyxy.tolist()[ri]
				sx1, sy1, sx2, sy2 = s_boxes[idx].xyxy.tolist()[rj]
				bx1, by1, bx2, by2 = max_box(p_img, px1, py1, px2, py2, sx1, sy1, sx2, sy2)
				p_upper_img = p_img[int(by1) : int((by1 + by2)/2),
				                    int(bx1) : int(bx2)]
				# 헬멧의 박스와 이미지 얻기(yolo model 활용)
				h_boxes, h_imgs = find_box(helmet_model, p_upper_img, goal = 'helmet')
				if len(h_boxes[0].cls.tolist()) != 0:
					print('YES********************')
					if h_boxes[0].cls.tolist()[0] == 0.0:
						hx1, hy1, hx2, hy2 = h_boxes[0].xyxy.tolist()[0]
						#hx1 += bx1; hy1 += by1; hx2 += bx1; hy2 += by1
						p_img = cv2.rectangle(p_img, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (255, 0, 0), 2)
						index += 1
					#elif h_boxes[0].cls.tolist()[0] == 1.0:
					#	hx1, hy1, hx2, hy2 = h_boxes[0].xyxy.tolist()[0]
					#	hx1 += bx1; hy1 += by1; hx2 += bx1; hy2 += by1
					#	p_img = cv2.rectangle(p_img, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (0, 0, 255), 2)
				
				result_img = cv2.rectangle(p_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
				iidd = int(pid)
				if not os.path.exists(f'./RESULTS/{folder_name}/images/{iidd}'):
					os.mkdir(f'./RESULTS/{folder_name}/images/{iidd}')
					
				else: pass
				
				
				resized_image = cv2.resize(result_img, (480, 240))
				
				idx_n = str(idx).zfill(4)
				cv2.imwrite(f'./RESULTS/{folder_name}/images/{iidd}/{idx_n}.jpg', resized_image)
				print("SAVE IMAGE")
				
				
				'''
				인도 주행 여부 확인
				'''
				# 0 = 인도에 대한 segment 정보
				r_seg = road_model(seg_img, classes = 0)
				for r in r_seg:
					# segment 박스 정보 가져오기
					r_box = r.boxes
					if (len(r_box) != 0):
						# original image의 너비와 높이 정보 가져오기
						h, w, _ = p_img.shape
							# mask 정보 가져오기
						segment_tensor = r.masks.masks
						segment_tensor_cpu = segment_tensor.cpu() # cpu로 이동시켜야 에러 안뜸...
						segment_array = segment_tensor_cpu.numpy() # 넘파이 배열로 변경
						mask_array = np.transpose(segment_array, (1, 2, 0)) # 텐서 뒤집기...(높이, 너비, 채널 -> 너비, 높이, 채널)
						mask_image = cv2.resize(mask_array, (h, w)) # 마스크 이미지 조정(이미지 사이즈에 맞게)

						# 일부 영역만 탐색(scooter 하단 1/4 지점)
						for i in range(int(sy1)+int((sy1+sy2)/4), int(sy2), 1):
							for j in range(int(sx1), int(sx2), 1):
								area += 1 # 탐색 픽셀수 증가
								# 마스크 이미지에서 해당 픽셀이 1인지 확인 후 c 1씩 추가
	#							if (mask_image[min(i, mask_image.shape[0] - 1)][min(j, mask_image.shape[1] - 1)] == 1).any():
								if np.any(mask_image[min(i, mask_image.shape[0] - 1)][min(j, mask_image.shape[1] - 1)] == 1):
									c += 1
		# 인도주행을 판단하는 임계값보다 높은 경우 인도주행(1)으로 설정
		print("******************", area, c)
		if (c) > 0: # 수정 가능한 임계값
			output = [list(item) for item in output]
			for o_i, tuple_output in enumerate (output):
				if tuple_output[0] == pid:
					output[o_i][3] = 1
		print(index)
		if index > 10: # 헬멧 쓰고 있음

			none_helmet = 0
		else:
			none_helmet = 1 # 헬멧 안 쓰고 있음
			
		output = [list(item) for item in output]
		for o_i, tuple_output in enumerate (output):
			if tuple_output[0] == pid:
				output[o_i][2] = none_helmet
				
		output = [tuple(item) for item in output]
		print("헬멧착용: ", output)
		
#		except: pass
	print(output)
						
				
	iidd = [int(p) for p in pid]
	with open(f'./RESULTS/{folder_name}/violations/{iidd}.txt', 'w') as file:
		for tuple_item in output:
			elements = tuple_item[1:]  # Exclude the first element
			line = ''.join(map(str, elements))
			line += f',{folder_name}'
			file.write(line + '\n')
		
	image_folder_path = f'./RESULTS/{folder_name}/images'
	
	image_files = sorted([file for file in os.listdir(image_folder_path) if file.endswith((".jpg", ".jpeg", ".png"))])
	
	# 삭제할 이미지 식별하기
	images_to_delete = []
		# 이미지 삭제
	delete_c = 0
	for i, image_file in enumerate(image_files, 1):
		if i % 3 != 0:
			os.remove(os.path.join(image_folder_path, image_file))
	

	
		



### max box 찾기
def max_box(p_img, px1, py1, px2, py2, sx1, sy1, sx2, sy2):
	max_width, max_height = 0, 0
	x1, y1, x2, y2 = 0, 0, 0, 0
    
	if px2 - px1 > max_width:  max_width = px2 - px1
	if py2 - py1 > max_height: max_height = py2 - py1
    
	if sx2 - sx1 > max_width:  max_width = sx2 - sx1
	if sy2 - sy1 > max_height: max_height = sy2 - sy1
    
	x1 = int(min(px1, sx1)) +1
	y1 = int(min(py1, sy1)) +1
	x2 = int(max(px2, sx2)) -1
	y2 = int(max(py2, sy2)) -1
    
	return x1, y1, x2, y2


### cosine 유사도 계산
def cosine_similarity(f_px, f_py, l_px, l_py, f_sx, f_sy,  l_sx, l_sy):
	u = (f_px - l_px, f_py - l_py)
	v = (f_sx - l_sx, f_sy - l_sy)
	dot_product = np.dot(u, v)
	norm1 = np.linalg.norm(u)
	norm2 = np.linalg.norm(v)
	similarity = dot_product / (norm1 * norm2)
    
	return similarity


### 유클리드안 거리 계산
def euclidean_distance(f_px, f_py, l_px, l_py, f_sx, f_sy,  l_sx, l_sy):
	fu = np.array([f_px, f_py])
	fv = np.array([f_sx, f_sy])
	fd = np.linalg.norm(fu - fv)

	lu = np.array([l_px, l_py])
	lv = np.array([l_sx, l_sy])
	ld = np.linalg.norm(lu - lv)

    
	return fd, ld


### IoU 계산
def IoU(x1, y1, x2, y2, B_x1, B_y1, B_x2, B_y2):
	area_A = (x2 - x1) * (y2 - y1)
	area_B = (B_x2 - B_x1) * (B_y2 - B_y1)
	x1_i = max(x1, B_x1)
	y1_i = max(y1, B_y1)
	x2_i = min(x2, B_x2)
	y2_i = min(y2, B_y2)
	if x2_i < x1_i or y2_i < y1_i:
		area_i = 0
	else:
		area_i = (x2_i - x1_i) * (y2_i - y1_i)
	iou = area_i / (area_A + area_B - area_i)
    
	return iou


### box 정보 찾기
def find_box(model, frame, goal):
	boxes, imgs = [], []
	if goal == 'person':
		results = model.track(frame, tracker = 'bytetrack.yaml', classes=0, conf=0.25)
	elif goal == 'helmet':
		results = model.predict(frame)
	else:
		results = model.track(frame, tracker = 'bytetrack.yaml', conf=0.25)
	for result in results:
		boxes.append(result.boxes)
		imgs.append(result.orig_img)
        
	return boxes, imgs
	
	
