#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from PIL import Image

import imagehash
import six
import shutil

"""
Demo of hashing
"""


def find_similar_images(userpaths, hashfuncs):
	def is_image(filename):
		f = filename.lower()
		return f.endswith('.png') or f.endswith('.jpg') or \
			f.endswith('.jpeg') or f.endswith('.bmp') or \
			f.endswith('.gif') or '.jpg' in f or f.endswith('.svg')

	image_filenames = []
	for userpath in userpaths:
		image_filenames += [os.path.join(userpath, path) for path in os.listdir(userpath) if is_image(path)]
	images = {}
	rotations = [0, 90, 180, 270]
	for img in sorted(image_filenames):
		hash = []
		image = Image.open(img)
		for rotation in rotations:
			image_rot = image.rotate(rotation)
			hash.append(str(hashfunc(image_rot)))
		# check if any element of the string is in any key of images dictionary
		if len(images.keys()) > 0:
			for key in images.keys():
				if any(h in key for h in hash):
					hash = key
				else:
					hash = ''.join(hash)
		else:
			hash = ''.join(hash)
		images[hash] = images.get(hash, []) + [img]
	output_root = "./images/output/"
	if os.path.exists(output_root):
		shutil.rmtree(output_root)
	os.mkdir(output_root)
	for hash, img_list in six.iteritems(images):
		print("{}: {}".format(hash,len(img_list)))
		if len(img_list) > 1:
			output_path = os.path.join(output_root, hash)
		else:
			output_path = os.path.join(output_root, "unknown")
		if not os.path.exists(output_path):
			os.mkdir(os.path.join(output_path))
		for img in img_list:
			shutil.copy(img, os.path.join(output_path, os.path.basename(img)))

if __name__ == '__main__':  # noqa: C901
	import os
	import sys

	def usage():
		sys.stderr.write("""SYNOPSIS: %s [ahash|phash|dhash|...] [<directory>]
Identifies similar images in the directory.
Method:
  ahash:          Average hash
  phash:          Perceptual hash
  dhash:          Difference hash
  whash-haar:     Haar wavelet hash
  whash-db4:      Daubechies wavelet hash
  colorhash:      HSV color hash
  crop-resistant: Crop-resistant hash
(C) Johannes Buchner, 2013-2017
""" % sys.argv[0])
		sys.exit(1)
	hashfuncs = []
	if len(sys.argv) < 2: usage()
	else:
		for i in range(1, len(sys.argv) - 1):
			hashmethod = sys.argv[i]
			if hashmethod == 'ahash':
				def hashfunc(img):
					return imagehash.average_hash(img, hash_size=10)
			elif hashmethod == 'phash':
				hashfunc = imagehash.phash
			elif hashmethod == 'dhash':
				def hashfunc(img):
					return imagehash.dhash(img, hash_size=2)
			elif hashmethod == 'whash-haar':
				hashfunc = imagehash.whash
			elif hashmethod == 'whash-db4':
				def hashfunc(img):
					return imagehash.whash(img, mode='db4')
			elif hashmethod == 'colorhash':
				def hashfunc(img):
					return imagehash.colorhash(img, binbits=4)
			elif hashmethod == 'crop-resistant':
				imagehash.crop_resistant_hash
			else:
				usage()
			hashfuncs.append(hashfunc)
		userpaths = sys.argv[len(sys.argv) -1:]
		find_similar_images(userpaths=userpaths, hashfuncs=hashfuncs)