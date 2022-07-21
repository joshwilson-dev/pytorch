################
#### Header ####
################

# Title: renames objects
# Author: Josh Wilson
# Date: 01-05-2022
# Description: 
# This script runs through the annotation files of the selected directory
# and rename objects based on com_sci

###############
#### Setup ####
###############

import os
import tkinter
from tkinter import filedialog
from tkinter import messagebox

#################
#### Content ####
#################

com_sci = {
    # "australian white ibis": "australianwhiteibis-aves-pelecaniformes-threskiornithidae-threskiornis-molucca-adult-unknown",
    # "royal spoonbill": "royalspoonbill-aves-pelecaniformes-threskiornithidae-platalea-regia-adult-unknown",
    # "white-faced heron": "whitefacedheron-aves-pelecaniformes-ardeidae-egretta-novaehollandiae-adult-unknown",
    "australian pelican": "australianpelican-aves-pelecaniformes-pelecanidae-pelecanus-conspicillatus-adult-unknown",
    # "black swan": "blackswan-aves-anseriformes-anatidae-cygnus-atratus-adult-unknown",
    # "australian pied oystercatcher": "piedoystercatcher-aves-charadriiformes-haematopodidae-haematopus-longirostris-adult-unknown"
    # "bird": "bird-aves-unknown-unknown-unknown-unknown-unknown-unknown"
    # "australian wood duck": "australianwoodduck-aves-anseriformes-anatidae-chenonetta-jubata-adult-unknown",
    # "silver gull": "silvergull-aves-charadriiformes-laridae-chroicocephalus-novaehollandiae-adult-unknown",
    # "australian pied oystercatcher": "piedoystercatcher-aves-charadriiformes-haematopodidae-haematopus-longirostris-adult-unknown",
    # "gull-billed tern": "gullbilledtern-aves-charadriiformes-laridae-gelochelidon-nilotica-adult-unknown",
    "bar-tailed godwit": "bartailedgodwit-aves-charadriiformes-scolopacidae-limosa-lapponica-adult-unknown",
    "whimbrel": "whimbrel-aves-charadriiformes-scolopacidae-numenius-phaeopus-adult-unknown",
    # "black-winged stilt": "piedstilt-aves-charadriiformes-recurvirostridae-himantopus-leucocephalus-adult-unknown",
    "great knot": "greatknot-aves-charadriiformes-scolopacidae-calidris-tenuirostris-adult-unknown"
    # "caspian tern": "caspiantern-aves-charadriiformes-laridae-hydroprogne-caspia-adult-unknown"
    # "masked lapwing": "maskedlapwing-aves-charadriiformes-charadriidae-vanellus-miles-adult-unknown",
    # "aves-charadriiformes-charadriidae-vanellus-miles-unkown-unknown": "maskedlapwing-aves-charadriiformes-charadriidae-vanellus-miles-adult-unknown"
    # "anas superciliosa": "pacificblackduck-aves-anseriformes-anatidae-anas-superciliosa-adult-unknown",
    # "vanellus miles": "maskedlapwing-aves-charadriiformes-charadriidae-vanellus-miles-adult-unknown",
    # "chenonetta jubata": "australianwoodduck-aves-anseriformes-anatidae-chenonetta-jubata-adult-unknown",
    # "gallinula tenebrosa": "duskymoorhen-aves-gruiformes-rallidae-gallinula-tenebrosa-adult-unknown",
    # "phalacrocorax sulcirostris": "littleblackcormorant-aves-suliformes-phalacrocoracidae-phalacrocorax-sulcirostris-adult-unknown"
    # "threskiornis spinicollis": "strawneckedibis-aves-pelecaniformes-threskiornithidae-threskiornis-spinicollis-adult-unknown"
    # "threskiornis molucca": "australianwhiteibis-aves-pelecaniformes-threskiornithidae-threskiornis-molucca-adult-unknown",
    # "bubulcus ibis": "cattleegret-aves-pelecaniformes-ardeidae-bubulcus-ibis-adult-unknown"
    # "aves-gruiformes-rallidae-porphyrio-melanotus-unknown-juvenile": "australasianswamphen-aves-gruiformes-rallidae-porphyrio-melanotus-juvenile-unknown",
    # "aves-gruiformes-rallidae-porphyrio-melanotus-unknown-adult": "australasianswamphen-aves-gruiformes-rallidae-porphyrio-melanotus-adult-unknown",
    # "aves-gruiformes-rallidae-gallinula-tenebrosa-unknown-adult": "duskymoorhen-aves-gruiformes-rallidae-gallinula-tenebrosa-adult-unknown",
}

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
        print ("You chose: %s" % tempdir)
    return tempdir

file_path_variable = search_for_file_path()

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to rename the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # iterate through files in dir
        for key, value in com_sci.items():
            for file in os.listdir():
                if file.endswith(".json"):
                    with open(file, 'r') as f:
                        filedata = f.read()
                        filedata = filedata.replace(key, value)
                        print(key, value)
                    with open(file, 'w') as f:
                        f.write(filedata)