import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for member in root.findall('object'):

                ymin, xmin, ymax, xmax = None, None, None, None

                for boxes in member.findall("bndbox"):
                    ymin = int(boxes.find("ymin").text)
                    xmin = int(boxes.find("xmin").text)
                    ymax = int(boxes.find("ymax").text)
                    xmax = int(boxes.find("xmax").text)

                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         xmin,
                         ymin,
                         xmax,
                         ymax)
                xml_list.append(value)
                
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('raccoon_labels.csv', index=None)
    print('Successfully converted xml to csv.')


main()
