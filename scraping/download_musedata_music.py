#!/usr/bin/env python2.7

"""
This script takes a list of URL's and downloads them to the given data directory.
The URLs correspond to .zip files of MuseData data. Each .zip file is one work by a composer, possibly with several
movements.
"""
import os
import re
import sys
from urllib2 import urlopen


def make_valid_filename(s):
    path = re.sub('[^\w\s-]', '', s).strip().lower()
    path = re.sub('[-\s]+', '-', path)
    return path


def main():
    usage = "download_musedata_music.py <index file> <destination directory>"
    if len(sys.argv) != 3:
        print usage
        exit()
    index_filename, data_dir = sys.argv[1:]
    with open(index_filename, "r") as index_file:
        for line in index_file.readlines():
            zip_url = line.strip()
            name_values = re.findall(r"composer=([a-z]+)&edition=(.+)&genre=(.+)&work=([0-9]+)", zip_url)
            composer, edition, genre, work = name_values[0]
            work_name = make_valid_filename("-".join([composer, edition, genre, work])) + ".zip"
            print "Downloading %s to %s" % (zip_url, os.path.join(data_dir, work_name))

            response = urlopen(zip_url)
            with open(work_name, 'wb') as f:
                while True:
                    chunk = response.read()
                    if not chunk:
                        break
                    f.write(chunk)


if __name__ == '__main__':
    main()
