from contextlib import contextmanager
from os.path import getsize, basename
from tqdm import tqdm

from memorize import Memorize
Memorize.USE_CURRENT_DIR = False

import csv


def build_csv_index(file_name):
    index = {}
    count = 0
    multiples = 0
    last_image_id = None
    print("Building img_id index for", file_name)
    with open(file_name, 'r') as csvfile:
        reader = PositionBasedCSVReader(csvfile, delimiter=',', quotechar='"')
        for f_position, row in reader:
            if row[0] != last_image_id:
                index[row[0]] = f_position
                last_image_id = row[0]
    print (file_name, "last row was", row)
    return index, last_image_id


def line_count(fn):
    return sum((1 for i in open(fn, 'rb')))


class it(object):

    def __init__(self, fd, pb):
        self.fd = fd
        self.pb = pb

    def __iter__(self):
        processed_bytes = 0
        for line in self.fd:
            processed_bytes += len(line)
            # update progress every MB.
            if processed_bytes >= 1024 * 1024:
                self.pb.update(processed_bytes)
                processed_bytes = 0

            yield line

        # finally
        self.pb.update(processed_bytes)
        self.pb.close()


@contextmanager
def pbopen(filename):
    total = getsize(filename)
    pb = tqdm(total=total, unit="B", unit_scale=True,
              desc=basename(filename), miniters=1, leave=False)
    # ncols=80, ascii=True)

    with open(filename, newline='') as fd:
        line_iterator = it(fd, pb)
        yield line_iterator


class PositionBasedCSVReader(object):
    """Like `csv.reader`, but yield successive pairs of:

    (
        <int> file position,
        <list> row,
    )
    """

    def __init__(self, csvfile, dialect='excel', **fmtparams):
        self.fp = csvfile
        self.dialect = dialect
        self.fmtparams = fmtparams
        self.line_iterator = iter(self.fp.readline, '')

    def __iter__(self):
        return self

    def seek(self, position):
        self.fp.seek(position)

    def _get_csv_row_from_line(self, line):
        return next(csv.reader([line], self.dialect, **self.fmtparams))

    def _get_next_row(self):
        line = next(self.line_iterator)
        return self._get_csv_row_from_line(line)

    def __next__(self):
        position = self.fp.tell()
        row = self._get_next_row()
        return position, row