import os, shutil, random
from PIL import Image

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


class SaigonBeer_Recognition:
  def __init__(self, cfg):
    self.cfg = cfg
    self.classifier = YOLO(self.cfg["Model"]['classifier'])
    self.detector = YOLO(self.cfg["Model"]['detector'])
    self.color = {#"others": (255,0,0),
                  "saigon_export":(0,255,0),
                  "saigon_chill":(0,0,255),
                  "saigon_large":(255,255,0),
                  "saigon_special":(255,0,255),
                  "saigon_gold":(255,0,0)}

  def forward(self, img, save_path=None):
    detected_objs = self.detector.predict(img)
    object_info = self.get_objects(detected_objs)
    object_info = self.box_filtering(object_info) # dictionary

    result = {}
    annotated_image = img.copy()
    for key, value in object_info.items():
      cropped_img = self.image_cropping(img, value['bbox'])
      kind = value['label']

      beer = self.classifier.predict(source=cropped_img, conf=0.5)
      beer = self.got_final_result(beer)

      result[key] = {"shape": kind, "beer": beer}
      if beer == "others":
        continue
    
      annotated_image = self.draw_img(annotated_image, value['bbox'], kind, beer, color=self.color[beer])
    return result, annotated_image

  def got_final_result(self, results):
    '''
    Get the label of the largest probability.
    Args:
        results: The results of the prediction.
    Returns:
      The label of the largest probability.
    '''
    id2label = results[0].names
    for result in results:
      cls = result.probs.top1
      return id2label[cls]

  def image_cropping(self, img, bbox):
    '''
    Crop the image based on the bounding box.
    Args:
        img: The path of the image.
        bbox: The bounding box of the object.
    Returns:
      The cropped image.
    '''
    x_min, y_min, x_max, y_max = bbox
    cropped_img = img.crop((x_min, y_min, x_max, y_max))
    return cropped_img

  def get_objects(self, predictions):
    '''
    Get the information of detected object.
    Args:
        predictions: The results of the prediction.
    Returns:
      A dictionary containing the information of detected object.
    '''
    object_dict = {}
    for i, prediction in enumerate(predictions):
      boxes = prediction.boxes
      for j, box in enumerate(boxes):
        object_key = f"object_{i}_{j}"
        object_dict[object_key] = {
            'label': prediction.names[int(box.cls)],
            'bbox': box.xyxy.tolist()[0],
            'prob': box.conf.tolist()[0]
        }
    return object_dict

  def box_filtering(self, predictions):
    '''
    Filtering out the lowest boxes
    Args:
        predictions: The results of the prediction.
    Returns:
      The filtered predictions.
    '''
    # Create a copy of the keys to iterate over
    keys_to_remove = []
    for key, value in predictions.items():
      if value['prob'] < 0.5:
        keys_to_remove.append(key)
    # Remove keys outside of the loop
    for key in keys_to_remove:
      del predictions[key]
    return predictions

  def draw_img(self, img, bbox, shape, beer, color=(0,255,0)):
    self.annotator = Annotator(img)
    self.annotator.box_label(bbox, f"{beer} {shape}", color=color)

    annotated_image = self.annotator.result()
    # Convert the NumPy array to a PIL Image before saving
    annotated_image = Image.fromarray(annotated_image) 

    return annotated_image