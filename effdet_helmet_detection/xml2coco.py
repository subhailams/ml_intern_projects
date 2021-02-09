#!/usr/bin/python

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET
import glob
import cv2

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#  "motorbike": 14, "person": 15, "pottedplant": 16,
#  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.
    
    Arguments:
        xml_files {list} -- A list of xml file paths.
    
    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    cnt = 0
    json_dict = {"categories": [],"images": [], "type": "instances", "annotations": []}

    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = xml_file.split('/')[4].split('.')[0] + '.jpg'
        if(cnt%100 == 0):
            print(cnt)
        cnt += 1
#         print(filename)
        image_id = filename
        img = cv2.imread('datasets/Dataset_nl/validation/Helmet/' + image_id)
        width = float(img.shape[1])
        height = float(img.shape[0])
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        
        cat = {"supercategory": "none", "id": 0, "name": "helmet"}
        json_dict["categories"].append(cat)
        
        for project in root.findall('project'):
            InstanceIDList = []
            for clas in project.findall('class'):
                category_id = clas.get('id')
                
                for box in clas.findall('boundingbox') or clas.findall('Polyline'):
                    bnd_id = box.get('id')
                    xmin = int(box.find('x1').text)
                    ymin = int(box.find('y1').text)
                    xmax = int(box.find('x2').text)
                    ymax = int(box.find('y2').text)
                    o_width = abs(xmax - xmin)
                    o_height = abs(ymax - ymin)
                    ann = {
                        "area": o_width * o_height,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": [xmin, ymin, o_width, o_height],
                        "category_id": category_id,
                        "id": bnd_id,
                        "ignore": 0,
                        "segmentation": [],
                    }
                    json_dict["annotations"].append(ann)
 


    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotation to COCO format."
    )
    parser.add_argument("xml_dir", help="Directory path to xml files.", type=str)
    parser.add_argument("json_file", help="Output COCO format json file.", type=str)
    args = parser.parse_args()
    xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))

    # If you want to do train/test split, you can pass a subset of xml files to convert function.
    print("Number of xml files: {}".format(len(xml_files)))
    convert(xml_files, args.json_file)
    print("Success: {}".format(args.json_file))
