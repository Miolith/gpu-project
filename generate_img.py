import cv2
import sys


def extract_frame(vid_path, output_folder):
    vidcap = cv2.VideoCapture(vid_path)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(output_folder + "frame%d.jpg" % count, image)     # save frame as JPEG file      
      success,image = vidcap.read()
      count += 1
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Invalid arguments, usage :\n./{sys.argv[0]} video_source_file output_folder")
    else:
        extract_frame(sys.argv[1], sys.argv[2])

