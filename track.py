from ultralytics import YOLO
import cv2
from typing import List
from collections import defaultdict
import numpy as np
import scipy.signal
from deep_sort_realtime.deepsort_tracker import DeepSort
import Ruya.Utils as ruya


# Track an objcet 
def object_location(image_width, objectX, object_width):

    image_width = image_width / 3
    object_center = objectX + (object_width / 2)
    if object_center < image_width:
        return 'يسار'
    elif object_center < image_width*2:
        return 'وسط'
    else:
        return 'يمين'

def which_direction(track):
    if len(track) < 2:
        return None
    start, end = np.array(track[:3])[:, 0], np.array(track[-3:])[:, 0]
    return 'right' if start.mean() < end.mean() else 'left'

def is_coming(track) -> bool:
    if len(track) < 2:
        return False

    (fx, fy, fw, fh), (lx, ly, lw, lh) = track[0], track[-1]

    initial_area = fw * fh
    final_area = lw * lh
    return final_area > initial_area * 1.02

def is_near(width, height, track):
    (x, y, w, h) = track[-1]
    return w*h >= width*height*0.02

def track():
    """Track when there's an object/person in front of a blind person by capturing frames at regular intervals and warn when the object/person is near them."""
    should_go = defaultdict(lambda: [False, False])
    model = YOLO('yolov5n.pt')
    
    cap = cv2.VideoCapture(0)
    
    # Store the track history
    track_history = defaultdict(list)
    deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)
    
    alert=None
    while cap.isOpened():
        
        should_go_left = should_go_right = is_person = False
        success, frame = cap.read()
        for _ in range(24):
            cap.read()

        if success:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            results = model.track(frame)[0] 
    
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
    
            indices = confidences >= 0.8
            filtered_boxes = boxes[indices]
            filtered_confidences = confidences[indices]
            filtered_classes = classes[indices]
    
            # process data in deepsort format 
            detections = []
            for box, conf, cls in zip(filtered_boxes, filtered_confidences, filtered_classes):
                cls=int(cls)
                x1, y1, x2, y2 = map(int, box[:4])
                bbox = [x1, y1, x2 - x1, y2 - y1]
                if cls == 0:
                    is_person=True
                detections.append((bbox, conf,cls))
            
            tracks = deepsort.update_tracks(detections, frame=frame)
    
    
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
    
                # get bbox 
                track_id = track.track_id
                bbox = track.to_tlbr() 
                x1, y1, x2, y2 = map(int, bbox)
                box = [x1 + (x2-x1) / 2, y1 + (y2-y1) / 2, x2-x1, y2-y1]
    
                # Update track history
                track_history[track_id].append(box)
    
                if is_coming(track_history[track_id]):
                    track_points = np.array(track_history[track_id])[:, :2]
                    
                    x = scipy.signal.medfilt(track_points[:, 0], kernel_size=5)
                    y = scipy.signal.medfilt(track_points[:, 1], kernel_size=5)
                    color = (0, 0, 255) if which_direction(track_history[track_id]) == 'right' else (0, 255, 0)
                    cv2.polylines(
                        frame,
                        [np.stack([x, y]).T[:, None, :].astype(np.int32)],
                        isClosed=False,
                        color=color,
                        thickness=10,
                    )
        
                    if is_near(width, height, track_history[track_id]):
                        if which_direction(track_history[track_id]) == 'right':
                            should_go[track_id][0] = True
                        else:
                            should_go[track_id][1] = True
                elif track_id in should_go:
                    should_go.clear()

            # if there are two object at front
            alert_left = alert_right = False
            for left, right in should_go.values():
                if left:
                    if not should_go_left:
                        should_go_left = True
                        alert_left = True
                elif right:
                    if not should_go_right:
                        should_go_right = True
                        alert_right = True

            if (alert_left and should_go_right) or (alert_right and should_go_left):
                alert='وقف'
            elif alert_left:
                alert='يسار'
            elif alert_right:
                alert='يمين'

            if alert == 'وقف':
               ruya.text_to_speech_stream( "توقف الآن، يوجد شخصان في طريقك.")
               alert=None
               print(len(should_go))
               should_go.clear()
            elif (alert != None) and is_person:
                ruya.text_to_speech_stream(f'انتبه، هناك شخص أمامك. يمكنك الذهاب الى {alert}')
                alert=None
                print(len(should_go))
                should_go.clear()
            elif (alert != None):
                ruya.text_to_speech_stream(f'انتبه، هناك شخص أمامك. يمكنك الذهاب الى {alert}')
                alert=None
                print(len(should_go))
                should_go.clear()
                
        else:
            break
    
    cap.release()

if __name__ == "__main__":
    track()
    
   
