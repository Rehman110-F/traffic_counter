from detection import  run_detection , video_setup
from pathlib import Path

video_path =  Path("input/input2.mp4")
output_path = Path("output/output_video2.mp4")
cap , out= video_setup(video_path , output_path)


run_detection(cap, out)