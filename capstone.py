import cv2, os, shutil, zipfile
from datetime import datetime
from methods.detect import search
#from methods.segment import segmentation

  

def work(video, folder_name, main_idx):
	search(video, folder_name, main_idx)
	
	
def image_mode(time, path, batch_size = 150):
	direc_setting(time)
    
	images = [img for img in sorted(os.listdir(path)) if img.endswith(".jpg")]
	image_count = 0
	for idx in range(len(images) // batch_size):
		batch_images = images[image_count:image_count + batch_size]
		video_name = f'RESULTS/{time}/videos/video_{idx+1}.mp4'
		to_video(path, batch_images, video_name, 0)
		image_count += batch_size
		work(video_name, time, idx)
		idx += 1
	save_zip(time)


def video_mode(time, path, batch_size = 150):
	direc_setting(time)
    
	video = cv2.VideoCapture(path)
	frame_count = 1; idx = 0
	image_list = []
	while True:
		ret, frame = video.read()
		if not ret: break
		if (frame_count % batch_size == 0):
			name = f'./RESULTS/{time}/video_{idx+1}.mp4'
			to_video('./RESULTS/'+time, image_list, f'./RESULTS/{time}/videos/video_{idx+1}.mp4', 1)
			idx += 1; image_list = []
			work(name, time, idx)
			
		image_list.append(frame)
		frame_count += 1
	save_zip(time)


def stream_mode(batch_size = 150):
	now = datetime.now(); folder_name = now.strftime("%Y-%m-%d__%H-%M-%S")
	direc_setting(folder_name)
    
	video_capture = cv2.VideoCapture(0)
	frame_count = 1; idx = 0
	image_list = []
	while True:
		ret, frame = video_capture.read()
		if cv2.waitKey(1) & 0xFF == ord('q'): break
		if (frame_count % batch_size == 0):
			to_video('./RESULTS/'+folder_name, image_list, f'./RESULTS/{folder_name}/videos/video_{idx+1}.mp4', 1)
			idx += 1; image_list = []
			work(f'./RESULTS/{folder_name}/video_{idx+1}.mp4', folder_name, idx)
		image_list.append(frame)
		frame_count += 1
	save_zip(time)


def direc_setting(folder_name):
    os.mkdir('./RESULTS/'+folder_name)
    
    if not os.path.exists('./RESULTS/'+folder_name+'/images/'): 
        os.mkdir('./RESULTS/'+folder_name+'/images/')
    if not os.path.exists('./RESULTS/'+folder_name+'/violations/'): 
        os.mkdir('./RESULTS/'+folder_name+'/violations/')
    if not os.path.exists('./RESULTS/'+folder_name+'/videos/'): 
        os.mkdir('./RESULTS/'+folder_name+'/videos/')

        
def to_video(path, image_list, video_name, mode):
    if mode == 0: frame = cv2.imread(os.path.join(path, image_list[0]))
    else: frame = image_list[0]
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))

    for image in image_list:
        if mode == 0: video.write(cv2.imread(path+'/'+image))
        elif mode == 1: video.write(image)

    cv2.destroyAllWindows()
    video.release()


def save_zip(time):
	folder_path_v = f'./RESULTS/{time}/violations'
	folder_path_i = f'./RESULTS/{time}/images'
	zip_file_path = f'./RESULTS/{time}/{time}.zip'

	zip_file = zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED)

	for root, _, files in os.walk(folder_path_v):
		for file in files:
		    file_path = os.path.join(root, file)
		    zip_file.write(file_path, os.path.join("violations", file))
		    
	for root, _, files in os.walk(folder_path_i):
		for file in files:
		    file_path = os.path.join(root, file)
		    zip_file.write(file_path, os.path.join("images", file))
		    
	zip_file.close()


while True:
	mode = int(input("Image[0], Video[1], Real-time[2]: "))
    
	if mode == 0 or mode == 1:
		time = input("Date and time( ex. 2011-01-01T00:00:00 ): ")
		path = input("Input path: ")
        
		if mode == 0: 
			image_mode(time, path)
			break
		else:
			video_mode(time, path)
			break
    
	elif mode == 2: stream_mode(); break
    
	else: print("Input again: ")
        
        
