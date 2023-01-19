from PIL import Image
from multiprocessing import Pool
import hashlib
import os
import itertools
import shutil
# size of boxes that has ~equal amount over overlap between the boxes
# and the amount of unsampled area
fiveboxsize = 0.46
fiveboxes = [
    (0.0, 0.0,  fiveboxsize,    fiveboxsize),
    (0.0, 1-fiveboxsize,  fiveboxsize,  1.0),
    (1-fiveboxsize, 0.0, 1.0,   fiveboxsize),
    (1-fiveboxsize, 1-fiveboxsize, 1.0, 1.0),
    (0.5-fiveboxsize/2.0, 0.5-fiveboxsize/2.0, 0.5+fiveboxsize/2.0, 0.5+fiveboxsize/2.0)
]

def colorHash(Img, colorspace=None):
    ''' Regrouped colour hashing function to avoid repeating code.'''

    # requested colorspace defaults to HSV
    cspace = colorspace if colorspace else 'HSV'
    # one box or five boxes requested
    boxes = fiveboxes

    # The image is not in the requested mode, convert
    if not Img.mode == cspace:
        Img = Img.convert(cspace)

    # resample to speed up the calculation
    size = 100
    Img = Img.resize((size, size), Image.Resampling.BOX)

    # split in bands
    channels = [ch.getdata() for ch in Img.split()]

    values = []
    # get measurements for each box
    for bx in boxes:
        # get a measurement for each channel
        for idx, ch in enumerate(channels):
            data = subImage(ch, bx)
            if cspace == 'HSV' and idx == 1:
                data = list(data)
                medianH = statsMedian(data)
                quant = statsQuantiles([(h-medianH+128) % 255 for h in data])
                values.append(round(medianH))
                values.append(round(quant[2] - quant[0]))
            else:
                quant = statsQuantiles(data)
                values.append(round(quant[1]))
                values.append(round(quant[2] - quant[0]))
    return values

def hsvHash(Img):
    return colorHash(Img, colorspace='HSV')

def subImage(Img, FracBox):
    'return a subImage from Image based on coordinates in dimensionless units'
    width, height = Img.size
    left = round(width*FracBox[0])
    right = round(width*FracBox[2])
    bottom = round(height*FracBox[1])
    top = round(height*FracBox[3])
    return Img.crop((left, bottom, right, top))

# not very careful but quick median function
def statsMedian(a):
    aa = sorted(a)
    return aa[round(len(aa)*0.50)]

# not very careful but quick 4-quantiles function
def statsQuantiles(a):
    aa = sorted(a)
    return [aa[round(len(aa)*i)] for i in [0.25, 0.5, 0.75]]

def match(limit, hashA, hashB):
	distArr = [
		abs(hashA[i]-hashB[i]) if i % 6
		else min((hashA[i]-hashB[i]) % 255, (hashB[i]-hashA[i]) % 255)
		for i in range(len(hashA))
		]
	val = sum(distArr)/len(distArr)
	return val

def find_similar_images(userpaths):
	def is_image(filename):
		f = filename.lower()
		return f.endswith('.png') or f.endswith('.jpg') or \
			f.endswith('.jpeg') or f.endswith('.bmp') or \
			f.endswith('.gif') or '.jpg' in f or f.endswith('.svg')

	image_filenames = []
	for userpath in userpaths:
		image_filenames += [os.path.join(userpath, path) for path in os.listdir(userpath) if is_image(path)]
	hashes = {"image": [], "hash": []}
	for img in sorted(image_filenames):
		hash = hsvHash(Image.open(img))
		hashes["hash"].append(hash)
		hashes["image"].append(img)
	hash_comb = list(itertools.combinations(hashes["hash"], 2))
	img_comb = list(itertools.combinations(hashes["image"], 2))
	distances = {"imageA": [], "imageB": [], "distance": []}
	results = dict.fromkeys(hashes["image"], ["cluster-0", 1000])
	cluster = 1
	for i in range(len(hash_comb)):
		hashA, hashB = hash_comb[i]
		imgA, imgB = img_comb[i]
		distance = match(4, hashA, hashB)
		# print(imgA, imgB, distance)
		distances["imageA"].append(imgA)
		distances["imageB"].append(imgB)
		distances["distance"].append(distance)
		if distance < 13:
			if results[imgA][0] != "cluster-0" and results[imgB][0] == "cluster-0":
				results[imgB] = [results[imgA][0], distance]
			elif results[imgB][0] != "cluster-0" and results[imgA][0] == "cluster-0":
				results[imgA] = [results[imgB][0], distance]
			elif results[imgA][0] != "cluster-0" and results[imgB][0] != "cluster-0":
				# combine clusters
				check = results[imgB][0]
				for key, val in results.items():
					# checking for required value
					if val[0] == check:
						results[key] = results[imgA]
			else:
				results[imgA] = ["cluster-" + str(cluster), distance]
				results[imgB] = ["cluster-" + str(cluster), distance]
				cluster += 1
	output_root = "./datasets/bird-mask/dataset/instances/australian_white_ibis/output"
	if os.path.exists(output_root):
		shutil.rmtree(output_root)
	os.mkdir(output_root)
	clusters = [a for a, b in results.values()]
	for cluster in set(clusters):
		os.mkdir(os.path.join(output_root, cluster))
	for img_pth, cluster in results.items():
		shutil.copy(img_pth, os.path.join(output_root, cluster[0], os.path.basename(img_pth)))

find_similar_images(["./datasets/bird-mask/dataset/instances/australian_white_ibis"])