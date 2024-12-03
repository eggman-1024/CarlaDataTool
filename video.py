import cv2

def main():    
    data_path = "./raw_data/record_2022_0726_1633/vehicle.tesla.model3.master/image_2/"
    fps = 5          # 视频帧率
    size = (1382, 512) # 需要转为视频的图片的尺寸
    video = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    
    for i in range(76,1075,3):      
        image_path = data_path + f"{i}".rjust(10,"0") + ".png"
        img = cv2.imread(image_path)
        video.write(img)
    
    video.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()