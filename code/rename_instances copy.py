################
#### Header ####
################

# Title: consolidate labelled images into dataset 
# Author: Josh Wilson
# Date: 01-06-2022
# Description: 
# This script runs through sub-dir of the selected directory
# looking for annotated image files and copying them to
# either a private or public dataset folder.

###############
#### Setup ####
###############

import os
import shutil
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import json

#################
#### Content ####
#################

# create function for user to select dir
root = tkinter.Tk()
root.withdraw()

def search_for_file_path ():
    currdir = os.getcwd()
    tempdir = filedialog.askdirectory(
        parent=root,
        initialdir=currdir,
        title='Please select a directory')
    if len(tempdir) > 0:
        print ('You chose: %s' % tempdir)
    return tempdir

file_path_variable = search_for_file_path()

label_dict = {
    "{'name': 'Lesser Black-backed Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'larus', 'species': 'fuscus', 'age': 'adult'}": "{'name': 'Lesser Black-backed Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'larus', 'species': 'fuscus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'European Herring Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'larus', 'species': 'argentatus', 'age': 'adult'}": "{'name': 'European Herring Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'larus', 'species': 'argentatus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'European Herring Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'larus', 'species': 'argentatus', 'age': 'juvenile'}": "{'name': 'European Herring Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'larus', 'species': 'argentatus', 'age': 'juvenile', 'pose': 'unknown'}",
    "{'name': 'Bird', 'class': 'aves', 'order': 'unknown', 'family': 'unknown', 'genus': 'unknown', 'species': 'unknown', 'age': 'adult'}": "{'name': 'Bird', 'class': 'aves', 'order': 'unknown', 'family': 'unknown', 'genus': 'unknown', 'species': 'unknown', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Lesser Black-backed Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'larus', 'species': 'fuscus', 'age': 'juvenile'}": "{'name': 'Lesser Black-backed Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'larus', 'species': 'fuscus', 'age': 'juvenile', 'pose': 'unknown'}",
    "{'name': 'Pied Oystercatcher', 'class': 'aves', 'order': 'charadriiformes', 'family': 'haematopodidae', 'genus': 'haematopus', 'species': 'longirostris', 'age': 'adult'}": "{'name': 'Pied Oystercatcher', 'class': 'aves', 'order': 'charadriiformes', 'family': 'haematopodidae', 'genus': 'haematopus', 'species': 'longirostris', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Mediterranean Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'ichthyaetus', 'species': 'melanocephalus', 'age': 'adult'}": "{'name': 'Mediterranean Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'ichthyaetus', 'species': 'melanocephalus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Mediterranean Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'ichthyaetus', 'species': 'melanocephalus', 'age': 'chick'}": "{'name': 'Mediterranean Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'ichthyaetus', 'species': 'melanocephalus', 'age': 'chick', 'pose': 'unknown'}",
    "{'name': 'Chinstrap Penguin', 'class': 'aves', 'order': 'sphenisciformes', 'family': 'spheniscidae', 'genus': 'pygoscelis', 'species': 'antarcticus', 'age': 'adult'}": "{'name': 'Chinstrap Penguin', 'class': 'aves', 'order': 'sphenisciformes', 'family': 'spheniscidae', 'genus': 'pygoscelis', 'species': 'antarcticus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Chinstrap Penguin', 'class': 'aves', 'order': 'sphenisciformes', 'family': 'spheniscidae', 'genus': 'pygoscelis', 'species': 'antarcticus', 'age': 'juvenile'}": "{'name': 'Chinstrap Penguin', 'class': 'aves', 'order': 'sphenisciformes', 'family': 'spheniscidae', 'genus': 'pygoscelis', 'species': 'antarcticus', 'age': 'juvenile', 'pose': 'unknown'}",
    "{'name': 'Gentoo Penguin', 'class': 'aves', 'order': 'spheniscidae', 'family': 'sphenisciformes', 'genus': 'pygoscelis', 'species': 'papua', 'age': 'adult'}": "{'name': 'Gentoo Penguin', 'class': 'aves', 'order': 'spheniscidae', 'family': 'sphenisciformes', 'genus': 'pygoscelis', 'species': 'papua', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Bird', 'class': 'aves', 'order': 'unknown', 'family': 'unknown', 'genus': 'unknown', 'species': 'unknown', 'age': 'unknown'}": "{'name': 'Bird', 'class': 'aves', 'order': 'unknown', 'family': 'unknown', 'genus': 'unknown', 'species': 'unknown', 'age': 'unknown', 'pose': 'unknown'}",
    "{'name': 'Chilean Flamingo', 'class': 'aves', 'order': 'phoenicopteriformes', 'family': 'phoenicopteridae', 'genus': 'phoenicopterus', 'species': 'chilensis', 'age': 'adult'}": "{'name': 'Chilean Flamingo', 'class': 'aves', 'order': 'phoenicopteriformes', 'family': 'phoenicopteridae', 'genus': 'phoenicopterus', 'species': 'chilensis', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Chilean Flamingo', 'class': 'aves', 'order': 'phoenicopteriformes', 'family': 'phoenicopteridae', 'genus': 'phoenicopterus', 'species': 'chilensis', 'age': 'juvenile'}": "{'name': 'Chilean Flamingo', 'class': 'aves', 'order': 'phoenicopteriformes', 'family': 'phoenicopteridae', 'genus': 'phoenicopterus', 'species': 'chilensis', 'age': 'juvenile', 'pose': 'unknown'}",
    "{'name': 'Australian Pelican', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'pelecanidae', 'genus': 'pelecanus', 'species': 'conspicillatus', 'age': 'adult'}": "{'name': 'Australian Pelican', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'pelecanidae', 'genus': 'pelecanus', 'species': 'conspicillatus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Australian Pelican', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'pelecanidae', 'genus': 'pelecanus', 'species': 'conspicillatus', 'age': 'juvenile'}": "{'name': 'Australian Pelican', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'pelecanidae', 'genus': 'pelecanus', 'species': 'conspicillatus', 'age': 'juvenile', 'pose': 'unknown'}",
    "{'name': 'Bird', 'class': 'aves', 'order': 'unknown', 'family': 'unknown', 'genus': 'unknown', 'species': 'unknown', 'age': 'juvenile'}": "{'name': 'Bird', 'class': 'aves', 'order': 'unknown', 'family': 'unknown', 'genus': 'unknown', 'species': 'unknown', 'age': 'juvenile', 'pose': 'unknown'}",
    "{'name': 'Welcome Swallow', 'class': 'aves', 'order': 'passeriformes', 'family': 'hirundinidae', 'genus': 'hirundo', 'species': 'neoxena', 'age': 'adult'}": "{'name': 'Welcome Swallow', 'class': 'aves', 'order': 'passeriformes', 'family': 'hirundinidae', 'genus': 'hirundo', 'species': 'neoxena', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Grey Teal', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'anas', 'species': 'gracilis', 'age': 'adult'}": "{'name': 'Grey Teal', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'anas', 'species': 'gracilis', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Little Grebe', 'class': 'aves', 'order': 'podicipediformes', 'family': 'podicipedidae', 'genus': 'tachybaptus', 'species': 'ruficollis', 'age': 'adult'}": "{'name': 'Little Grebe', 'class': 'aves', 'order': 'podicipediformes', 'family': 'podicipedidae', 'genus': 'tachybaptus', 'species': 'ruficollis', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Eurasian Coot', 'class': 'aves', 'order': 'gruiformes', 'family': 'rallidae', 'genus': 'fulica', 'species': 'atra', 'age': 'adult'}": "{'name': 'Eurasian Coot', 'class': 'aves', 'order': 'gruiformes', 'family': 'rallidae', 'genus': 'fulica', 'species': 'atra', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Pacific Black Duck', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'anas', 'species': 'superciliosa', 'age': 'adult'}": "{'name': 'Pacific Black Duck', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'anas', 'species': 'superciliosa', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Hardhead', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'aythya', 'species': 'australis', 'age': 'adult'}": "{'name': 'Hardhead', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'aythya', 'species': 'australis', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Black Swan', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'cygnus', 'species': 'atratus', 'age': 'adult'}": "{'name': 'Black Swan', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'cygnus', 'species': 'atratus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Pink-eared Duck', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'malacorhynchus', 'species': 'membranaceus', 'age': 'adult'}": "{'name': 'Pink-eared Duck', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'malacorhynchus', 'species': 'membranaceus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Black-necked Stilt', 'class': 'aves', 'order': 'charadriiformes', 'family': 'recurvirostridae', 'genus': 'himantopus', 'species': 'mexicanus', 'age': 'adult'}": "{'name': 'Black-necked Stilt', 'class': 'aves', 'order': 'charadriiformes', 'family': 'recurvirostridae', 'genus': 'himantopus', 'species': 'mexicanus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Gull-billed Tern', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'gelochelidon', 'species': 'nilotica', 'age': 'adult'}": "{'name': 'Gull-billed Tern', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'gelochelidon', 'species': 'nilotica', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Caspian Tern', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'hydroprogne', 'species': 'caspia', 'age': 'adult'}": "{'name': 'Caspian Tern', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'hydroprogne', 'species': 'caspia', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Great Knot', 'class': 'aves', 'order': 'charadriiformes', 'family': 'scolopacidae', 'genus': 'calidris', 'species': 'tenuirostris', 'age': 'adult'}": "{'name': 'Great Knot', 'class': 'aves', 'order': 'charadriiformes', 'family': 'scolopacidae', 'genus': 'calidris', 'species': 'tenuirostris', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Pied Stilt', 'class': 'aves', 'order': 'charadriiformes', 'family': 'recurvirostridae', 'genus': 'himantopus', 'species': 'leucocephalus', 'age': 'adult'}": "{'name': 'Pied Stilt', 'class': 'aves', 'order': 'charadriiformes', 'family': 'recurvirostridae', 'genus': 'himantopus', 'species': 'leucocephalus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Bar-tailed Godwit', 'class': 'aves', 'order': 'charadriiformes', 'family': 'scolopacidae', 'genus': 'limosa', 'species': 'lapponica', 'age': 'adult'}": "{'name': 'Bar-tailed Godwit', 'class': 'aves', 'order': 'charadriiformes', 'family': 'scolopacidae', 'genus': 'limosa', 'species': 'lapponica', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Whimbrel', 'class': 'aves', 'order': 'charadriiformes', 'family': 'scolopacidae', 'genus': 'numenius', 'species': 'phaeopus', 'age': 'adult'}": "{'name': 'Whimbrel', 'class': 'aves', 'order': 'charadriiformes', 'family': 'scolopacidae', 'genus': 'numenius', 'species': 'phaeopus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Royal Spoonbill', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'platalea', 'species': 'regia', 'age': 'adult'}": "{'name': 'Royal Spoonbill', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'platalea', 'species': 'regia', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'White-faced Heron', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'ardeidae', 'genus': 'egretta', 'species': 'novaehollandiae', 'age': 'adult'}": "{'name': 'White-faced Heron', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'ardeidae', 'genus': 'egretta', 'species': 'novaehollandiae', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Australian White Ibis', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'threskiornis', 'species': 'molucca', 'age': 'adult'}": "{'name': 'Australian White Ibis', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'threskiornis', 'species': 'molucca', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Masked Lapwing', 'class': 'aves', 'order': 'charadriiformes', 'family': 'charadriidae', 'genus': 'vanellus', 'species': 'miles', 'age': 'adult'}": "{'name': 'Masked Lapwing', 'class': 'aves', 'order': 'charadriiformes', 'family': 'charadriidae', 'genus': 'vanellus', 'species': 'miles', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Australian Wood Duck', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'chenonetta', 'species': 'jubata', 'age': 'adult'}": "{'name': 'Australian Wood Duck', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'chenonetta', 'species': 'jubata', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Silver Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'chroicocephalus', 'species': 'novaehollandiae', 'age': 'adult'}": "{'name': 'Silver Gull', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'chroicocephalus', 'species': 'novaehollandiae', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Cattle Egret', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'ardeidae', 'genus': 'bubulcus', 'species': 'ibis', 'age': 'adult'}": "{'name': 'Cattle Egret', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'ardeidae', 'genus': 'bubulcus', 'species': 'ibis', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Dusky Moorhen', 'class': 'aves', 'order': 'gruiformes', 'family': 'rallidae', 'genus': 'gallinula', 'species': 'tenebrosa', 'age': 'adult'}": "{'name': 'Dusky Moorhen', 'class': 'aves', 'order': 'gruiformes', 'family': 'rallidae', 'genus': 'gallinula', 'species': 'tenebrosa', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Little Black Cormorant', 'class': 'aves', 'order': 'suliformes', 'family': 'phalacrocoracidae', 'genus': 'phalacrocorax', 'species': 'sulcirostris', 'age': 'adult'}": "{'name': 'Little Black Cormorant', 'class': 'aves', 'order': 'suliformes', 'family': 'phalacrocoracidae', 'genus': 'phalacrocorax', 'species': 'sulcirostris', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Straw-necked Ibis', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'threskiornis', 'species': 'spinicollis', 'age': 'adult'}": "{'name': 'Straw-necked Ibis', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'threskiornis', 'species': 'spinicollis', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Australasian Swamphen', 'class': 'aves', 'order': 'gruiformes', 'family': 'rallidae', 'genus': 'porphyrio', 'species': 'melanotus', 'age': 'adult'}": "{'name': 'Australasian Swamphen', 'class': 'aves', 'order': 'gruiformes', 'family': 'rallidae', 'genus': 'porphyrio', 'species': 'melanotus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Magpie Lark', 'class': 'aves', 'order': 'passeriformes', 'family': 'monarchidae', 'genus': 'grallina', 'species': 'cyanoleuca', 'age': 'adult'}": "{'name': 'Magpie Lark', 'class': 'aves', 'order': 'passeriformes', 'family': 'monarchidae', 'genus': 'grallina', 'species': 'cyanoleuca', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Muscovy Duck', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'cairina', 'species': 'moschata', 'age': 'adult'}": "{'name': 'Muscovy Duck', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'cairina', 'species': 'moschata', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Torresian Crow', 'class': 'aves', 'order': 'passeriformes', 'family': 'corvidae', 'genus': 'corvus', 'species': 'orru', 'age': 'adult'}": "{'name': 'Torresian Crow', 'class': 'aves', 'order': 'passeriformes', 'family': 'corvidae', 'genus': 'corvus', 'species': 'orru', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Domestic Duck', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'anas', 'species': 'platyrhynchos domesticus', 'age': 'adult'}": "{'name': 'Domestic Duck', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'anas', 'species': 'platyrhynchos domesticus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Pied Butcherbird', 'class': 'aves', 'order': 'passeriformes', 'family': 'artamidae', 'genus': 'cracticus', 'species': 'nigrogularis', 'age': 'adult'}": "{'name': 'Pied Butcherbird', 'class': 'aves', 'order': 'passeriformes', 'family': 'artamidae', 'genus': 'cracticus', 'species': 'nigrogularis', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Australian Magpie', 'class': 'aves', 'order': 'passeriformes', 'family': 'artamidae', 'genus': 'gymnorhina', 'species': 'tibicen', 'age': 'adult'}": "{'name': 'Australian Magpie', 'class': 'aves', 'order': 'passeriformes', 'family': 'artamidae', 'genus': 'gymnorhina', 'species': 'tibicen', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Bird', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'anas', 'species': 'unknown', 'age': 'adult'}": "{'name': 'Bird', 'class': 'aves', 'order': 'anseriformes', 'family': 'anatidae', 'genus': 'anas', 'species': 'unknown', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Little Corella', 'class': 'aves', 'order': 'psittaciformes', 'family': 'cacatuidae', 'genus': 'cacatua licmetis', 'species': 'sanguinea', 'age': 'adult'}": "{'name': 'Little Corella', 'class': 'aves', 'order': 'psittaciformes', 'family': 'cacatuidae', 'genus': 'cacatua licmetis', 'species': 'sanguinea', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'African Openbill', 'class': 'aves', 'order': 'ciconiiformes', 'family': 'ciconiidae', 'genus': 'anastomus', 'species': 'lamelligerus', 'age': 'adult'}": "{'name': 'African Openbill', 'class': 'aves', 'order': 'ciconiiformes', 'family': 'ciconiidae', 'genus': 'anastomus', 'species': 'lamelligerus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'African Spoonbill', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'platalea', 'species': 'alba', 'age': 'adult'}": "{'name': 'African Spoonbill', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'platalea', 'species': 'alba', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Reed Cormorant', 'class': 'aves', 'order': 'suliformes', 'family': 'phalacrocoracidae', 'genus': 'microcarbo', 'species': 'africanus', 'age': 'adult'}": "{'name': 'Reed Cormorant', 'class': 'aves', 'order': 'suliformes', 'family': 'phalacrocoracidae', 'genus': 'microcarbo', 'species': 'africanus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'African Darter', 'class': 'aves', 'order': 'suliformes', 'family': 'anhingidae', 'genus': 'anhinga', 'species': 'rufa', 'age': 'adult'}": "{'name': 'African Darter', 'class': 'aves', 'order': 'suliformes', 'family': 'anhingidae', 'genus': 'anhinga', 'species': 'rufa', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Bird', 'class': 'aves', 'order': 'unknown', 'family': 'unknown', 'genus': 'unknown', 'species': 'unknown', 'age': 'chick'}": "{'name': 'Bird', 'class': 'aves', 'order': 'unknown', 'family': 'unknown', 'genus': 'unknown', 'species': 'unknown', 'age': 'chick', 'pose': 'unknown'}",
    "{'name': 'Yellow-billed Stork', 'class': 'aves', 'order': 'ciconiiformes', 'family': 'ciconiidae', 'genus': 'mycteria', 'species': 'ibis', 'age': 'adult'}": "{'name': 'Yellow-billed Stork', 'class': 'aves', 'order': 'ciconiiformes', 'family': 'ciconiidae', 'genus': 'mycteria', 'species': 'ibis', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Yellow-billed Stork', 'class': 'aves', 'order': 'ciconiiformes', 'family': 'ciconiidae', 'genus': 'mycteria', 'species': 'ibis', 'age': 'chick'}": "{'name': 'Yellow-billed Stork', 'class': 'aves', 'order': 'ciconiiformes', 'family': 'ciconiidae', 'genus': 'mycteria', 'species': 'ibis', 'age': 'chick', 'pose': 'unknown'}",
    "{'name': 'African Darter', 'class': 'aves', 'order': 'suliformes', 'family': 'anhingidae', 'genus': 'anhinga', 'species': 'rufa', 'age': 'chick'}": "{'name': 'African Darter', 'class': 'aves', 'order': 'suliformes', 'family': 'anhingidae', 'genus': 'anhinga', 'species': 'rufa', 'age': 'chick', 'pose': 'unknown'}",
    "{'name': 'African Spoonbill', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'platalea', 'species': 'alba', 'age': 'chick'}": "{'name': 'African Spoonbill', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'platalea', 'species': 'alba', 'age': 'chick', 'pose': 'unknown'}",
    "{'name': 'African Sacred Ibis', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'threskiornis', 'species': 'aethiopicus', 'age': 'adult'}": "{'name': 'African Sacred Ibis', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'threskiornithidae', 'genus': 'threskiornis', 'species': 'aethiopicus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Pied Avocet', 'class': 'aves', 'order': 'charadriiformes', 'family': 'recurvirostridae', 'genus': 'recurvirostra', 'species': 'avosetta', 'age': 'adult'}": "{'name': 'Pied Avocet', 'class': 'aves', 'order': 'charadriiformes', 'family': 'recurvirostridae', 'genus': 'recurvirostra', 'species': 'avosetta', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Common Tern', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'sterna', 'species': 'hirundo', 'age': 'adult'}": "{'name': 'Common Tern', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'sterna', 'species': 'hirundo', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Chaco Eagle', 'class': 'aves', 'order': 'accipitriformes', 'family': 'accipitridae', 'genus': 'buteogallus', 'species': 'coronatus', 'age': 'adult'}": "{'name': 'Chaco Eagle', 'class': 'aves', 'order': 'accipitriformes', 'family': 'accipitridae', 'genus': 'buteogallus', 'species': 'coronatus', 'age': 'adult', 'pose': 'unknown'}",
    "{'name': 'Aleutian Tern', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'onychoprion', 'species': 'aleuticus', 'age': 'adult'}": "{'name': 'Aleutian Tern', 'class': 'aves', 'order': 'charadriiformes', 'family': 'laridae', 'genus': 'onychoprion', 'species': 'aleuticus', 'age': 'adult', 'pose': 'unknown'}"
}

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        'CONFIRM',
        'Are you sure you want to rename the instances in:\n' + file_path_variable)
    if check =='yes':
        os.chdir(file_path_variable)
        # iterate through files in dir
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.endswith('.json'):
                    annotation_path = os.path.join(root, file)
                    annotation = json.load(open(annotation_path))
                    print('Renaming instances in', file)
                    for i in range(len(annotation['shapes'])):
                        old_label = annotation['shapes'][i]['label']
                        try:
                            new_label = label_dict[old_label]
                        except:
                            print(old_label, 'not in label dictionary')
                            continue
                        annotation['shapes'][i]['label'] = new_label
                    annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                    with open(annotation_path, 'w') as annotation_file:
                        annotation_file.write(annotation_str)