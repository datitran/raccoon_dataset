import pandas as pd
import argparse
from collections import namedtuple
from tqdm import tqdm
import os


def __split(df, group):
   data = namedtuple('data', ['filename', 'object'])
   gb = df.groupby(group)
   return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def yolo_txt_from_csv(input_csv, input_names, output_dir):
   with open(input_names, "r") as file:
      names = file.read().split('\n')
   df = pd.read_csv(input_csv)

   grouped = __split(df, 'filename')

   for group in tqdm(grouped, desc='groups'):
      filename = group.filename
      xs = []
      ys = []
      widths = []
      heights = []
      classes = []

      for _, row in group.object.iterrows():
         if not set(['class', 'width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']).issubset(
             set(row.index)):
            pass

         img_width = row['width']
         img_height = row['height']

         xmin = row['xmin']
         ymin = row['ymin']
         xmax = row['xmax']
         ymax = row['ymax']

         xs.append(round(xmin / img_width, 5))
         ys.append(round(ymin / img_height, 5))
         widths.append(round((xmax - xmin) / img_width, 5))
         heights.append(round((ymax - ymin) / img_height, 5))
         classes.append(row['class'])

      txt_filename = os.path.splitext(filename)[0] + '.txt'

      with open(os.path.join(output_dir, txt_filename), 'w+') as f:
         for i in range(len(classes)):
            f.write('{} {} {} {} {}\n'.format(names.index(classes[i]),
                                              xs[i],
                                              ys[i],
                                              widths[i],
                                              heights[i]))


if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description=
       'Reads the contents of a CSV file, containing object annotations and their corresponding images\'s dimensions, and generates TXT files for use with darknet and YOLOv3'
   )
   parser.add_argument('input_csv',
                       metavar='input_csv',
                       type=str,
                       help='Path to the input CSV file')
   parser.add_argument(
       'input_names',
       metavar='input_names',
       type=str,
       help='Path to the input .names file used by darknet, containing names of object classes')
   parser.add_argument(
       'output_dir',
       metavar='output_dir',
       type=str,
       help='Directory where the .txt output files will be created, one for each image contained in the CSV fle'
   )

   args = parser.parse_args()

   yolo_txt_from_csv(args.input_csv, args.input_names, args.output_dir)
