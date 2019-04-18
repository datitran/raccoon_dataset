import pandas as pd
import argparse
from collections import namedtuple
from tqdm import tqdm
import os


def __split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


def yolo_txt_from_csv(input_file, output_dir):
    df = pd.read_csv(input_file)

    grouped = __split(df, 'filename')

    for group in tqdm(grouped, desc='groups'):
        filename = group.filename
        xs = []
        ys = []
        widths = []
        heights = []
        classes = []

        for _, row in group.object.iterrows():
            if not set([
                    'class', 'width', 'height', 'xmin', 'xmax', 'ymin', 'ymax'
            ]).issubset(set(row.index)):
                pass

            img_width = row['width']
            img_height = row['height']

            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']

            xs.append(xmin / img_width)
            ys.append(ymin / img_height)
            widths.append((xmax - xmin) / img_width)
            heights.append((ymax - xmin) / img_height)
            classes.append(row['class'])

        txt_filename = os.path.splitext(filename)[0] + '.txt'

        with open(os.path.join(output_dir, txt_filename), 'w+') as f:
            for i in range(len(classes)):
                f.write('{} {} {} {} {}\n'.format(classes[i], xs[i], ys[i],
                                                  widths[i], heights[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Reads the contents of a CSV file, containing object annotations and their corresponding images\'s dimensions, and generates TXT files for use with darknet and YOLOv3'
    )
    parser.add_argument('input_file',
                        metavar='input_file',
                        type=str,
                        help='Path to the input CSV file')
    parser.add_argument(
        'output_dir',
        metavar='output_dir',
        type=str,
        help=
        'Directory where the .txt output files will be created, one for each image contained in the CSV fle'
    )

    args = parser.parse_args()

    yolo_txt_from_csv(args.input_file, args.output_dir)