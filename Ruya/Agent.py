import Ruya.Utils as ruya
import json
import chime
import os


chime.theme('big-sur')

with open('Ruya/location_data.json', 'r') as openfile:
    try:
        # Try to load the JSON data
        places = json.load(openfile)
        if not places:
            places = {}
    except json.JSONDecodeError:
        # If there's a JSONDecodeError, it means the file is empty or not properly formatted
        places = {}

# NLP imports 
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import  FunctionTool
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai 
openai.api_key = os.environ.get('OPENAI_API_KEY')


# NLP models 
processor_captioning = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_captioning = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()
llm = OpenAI(model="gpt-4o")


# Computer vision imports
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import torch

# CV models 
model_world = YOLO('yolov8s-world.pt')

def save_data():
    json_object = json.dumps(places, indent=4)
    # Writing to location_data
    with open("Ruya/location_data.json", "w") as outfile:
        outfile.write(json_object)

def summarize_video(**kwargs) -> str:
    """Explain in detail what is seen in front for a blind person by capturing and describing frames at regular intervals. be clear when explain. return the summary in arabic"""
    num_frames = 3
    cap = cv2.VideoCapture(0)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return "No frames in video device."

    frame_interval = frame_count // num_frames
    descriptions = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if not ret:
            continue  # Skip if frame is not captured successfully

        try:
            raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except cv2.error as e:
            print(f"Error converting frame: {e}")
            continue  # Skip if there is an error in conversion

        text = "There is front of me "
        inputs = processor_captioning(raw_image, text, return_tensors="pt")

        with torch.no_grad():
            outputs = model_captioning.generate(
                inputs['pixel_values'],
                max_length=120,
                num_beams=5,
                temperature=0.5,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True
            )

        description_image = processor_captioning.decode(outputs[0], skip_special_tokens=True)
        descriptions.append(description_image)

    cap.release()

    video_summary = " ".join(descriptions)
    return video_summary

def estimate_distance(object_name: str) -> str:
    '''Capture an image from the webcam and use the object name for input in this tool from user and measure the distence and return the distence to the user and the direction of the object from the tool in arabic only don't use english '''
    object_name_eng = ruya.ar_eng(object_name)

    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    model_world.set_classes([object_name_eng])

    results = model_world(img)[0]
    print(f'eng {object_name_eng}')
    print(f'object{object_name}')
    
    boxes = results.boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        return f'عذرًا، لم أتمكن من العثور عليه.'
    
    #return nearest object
    boxes = ruya.nearest_object(boxes)
    
    if boxes.shape[0]==1:
        boxes = boxes[0]
    
    x1, y1, x2, y2 = map(int, boxes)
    w = x2 - x1
    
    width = img.shape[1]
    

    direction = ruya.object_location(width,x1,w)
    
    estimated_distance = ruya.estimate_object_distance(np.array([x1, y1, x2, y2]))
    
    cap.release()

    if estimated_distance is not None:
        distance_text = f' يبعد عنك {ruya.meters_to_steps(estimated_distance)} خطوات '
        direction_text = f'وهو في جهة {direction}'

    return direction_text , distance_text

def save_location(object_name: str) -> str:
    '''save the location of an object when asked and use the object name and return the location in arabic'''
    object_name_eng = ruya.ar_eng(object_name)

    ruya.text_to_speech_stream(f'وين مكان {object_name}')
    chime.success()

    location = ruya.audio_transcription()
    where=f'حفظت {object_name} في {location}'

    places[object_name_eng]=location
    save_data()
    return where

def where_location(object_name: str) -> str:
    '''this tool for search if the a location of object stored before if not it will search for the object by using webcam capture and use the object name from user's input and return the location in arabic'''
    object_name_eng = ruya.ar_eng(object_name)

    if object_name_eng in places:
        location = places[object_name_eng]
    else:
        location = estimate_distance(object_name)

    return location

def get_agent():
    summarize_video_tool = FunctionTool.from_defaults(summarize_video)
    estimate_distance_tool = FunctionTool.from_defaults(estimate_distance)
    save_location_tool = FunctionTool.from_defaults(save_location)
    where_location_tool = FunctionTool.from_defaults(where_location)
    agent = ReActAgent.from_tools([estimate_distance_tool,where_location_tool, save_location_tool, summarize_video_tool], llm=llm, verbose=True)
    
    return agent