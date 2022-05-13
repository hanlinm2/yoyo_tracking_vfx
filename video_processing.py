import cv2 as cv 
import numpy as np

# Modify the variables below to select behavior
save_tracking_video = True
save_effect_video = True
num_effect_frames = 20 # how long the trail is
radius = 10
hue = 0 # Hue is between 0 and 179, could set to None
saturation = 150 # Saturation is between 0 and 255
num_interpolation = 5 # number of frames to fill in the gap between frames
start_frame = 100 # the first frame that will be used to select Region of Interst (ROI)

# Read input video
video = cv.VideoCapture("Videos/quad3.mov")
fps = video.get(cv.CAP_PROP_FPS) 
print("FPS:",fps)
for i in range(start_frame):
    isTrue, frame = video.read()

# Tracker and bounding box initialization
tracker = cv.TrackerCSRT_create()
bbox = cv.selectROI(frame, False)
ok = tracker.init(frame, bbox)
yoyo_locations = [(-1,-1)]*num_effect_frames # stores the (x,y) location of center of yoyo in a tuple.

# Set up output video
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
if save_tracking_video:
    tracked_video = cv.VideoWriter('Videos/tracking_video.mp4', cv.VideoWriter_fourcc(*'mp4v'),fps, size)
if save_effect_video:
    effect_video = cv.VideoWriter('Videos/effect_video.mp4', cv.VideoWriter_fourcc(*'mp4v'),fps, size)

# Read each frame
while video.isOpened():

    isTrue, frame = video.read()
    if isTrue:

        ok,bbox=tracker.update(frame)
        effect_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        

        if ok:
            (x,y,w,h)=[int(v) for v in bbox]
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            yoyo_locations.append((x+w//2, y + h//2))
            if len(yoyo_locations) == num_effect_frames + 1:
                yoyo_locations.pop(0)
        else:
            yoyo_locations.append((-1,-1))
            if len(yoyo_locations) == num_effect_frames + 1:
                yoyo_locations.pop(0)
        
        # insert more locations in between frames to make it more continuous
        interpolated_yoyo_locations = []
        for i, location in enumerate(yoyo_locations):
            if location ==  (-1,-1): continue

            # find next valid location that is not (-1,-1)
            j = i+1
            if j >= len(yoyo_locations): break
            while j < len(yoyo_locations) and yoyo_locations[j] == (-1,-1) :
                j += 1
            if j >= len(yoyo_locations): break
            
            cur_x, cur_y = location
            next_x, next_y = yoyo_locations[j]

            for step in range(num_interpolation+1):
                alpha = step * (1/num_interpolation)
                interpolated_x = int((1-alpha)*cur_x + (alpha * next_x))
                interpolated_y = int((1-alpha)*cur_y + (alpha * next_y))
                interpolated_yoyo_locations.append((interpolated_x, interpolated_y))

        # calucate the trail mask

        trail_mask = np.full((frame_height, frame_width), False)
        circle_mask = np.full((frame_height, frame_width), False)

        cur_radius = 1
        for cx, cy in interpolated_yoyo_locations:
            if cx == -1 or cy == -1: continue
            Y, X = np.ogrid[:frame_height, :frame_width]
            dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
            circle_mask = dist_from_center <= cur_radius
            if cur_radius < radius:
                cur_radius += 1
            trail_mask = np.logical_or(trail_mask, circle_mask)

        # edit the pixel values
        if hue != None:
            effect_frame[trail_mask, 0] = hue
        if saturation != None:
            effect_frame[trail_mask, 1] = saturation
    
        # make sure the range of values is between 0 and 225
        effect_frame.clip(0,255)

        # show the video frames
        cv.imshow('Tracking Video', frame)
        effect_frame = cv.cvtColor(effect_frame, cv.COLOR_HSV2BGR)
        cv.imshow('Effect Video', effect_frame)

        if cv.waitKey(20) & 0xFF==ord('q'):
            break 
        
        # save the video frames
        if save_tracking_video:
            tracked_video.write(frame)   

        if save_effect_video:
            effect_video.write(effect_frame)        
    else:
        break

video.release()
if save_tracking_video:  
    tracked_video.release()
if save_effect_video:
    effect_video.release()
cv.destroyAllWindows()