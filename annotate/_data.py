"""
Data processing functions that are cached, so we want to modify these infrequently
"""

from annotate.utils import PositionBasedCSVReader, build_csv_index
from memorize import Memorize
Memorize.USE_CURRENT_DIR = False

@Memorize
def count_class(file_name, class_label):
    count = 0
    multiples = 0
    images_seen = {}
    with open(file_name, 'r') as csvfile:
        reader = PositionBasedCSVReader(csvfile, delimiter=',', quotechar='"')
        last_image_id = None
        since_last = 0

        for f_position, row in reader:
            # if the label matches that which we are looking for, and the confidence is non-zero
            # (zero is for human-verified absence)
            if class_label == row[2] and row[3] != '0':
                since_last += 1
                if row[0] != last_image_id and row[0] not in images_seen:
                    images_seen[row[0]] = f_position
                    count += 1
                    last_image_id = row[0]
                    since_last = 1
                elif since_last == 2:
                    multiples += 1
    return images_seen, count, multiples

