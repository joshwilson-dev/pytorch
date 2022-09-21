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
"Little Black Cormorant_adult_suliformes_phalacrocoracidae_phalacrocorax_sulcirostris": "Little Black Cormorant_aves_adult_suliformes_phalacrocoracidae_phalacrocorax_sulcirostris",
"African Darter_adult_suliformes_anhingidae_anhinga_rufa": "African Darter_aves_adult_suliformes_anhingidae_anhinga_rufa",
"African Darter_chick_suliformes_anhingidae_anhinga_rufa": "African Darter_aves_chick_suliformes_anhingidae_anhinga_rufa",
"African Openbill_adult_ciconiiformes_ciconiidae_anastomus_lamelligerus": "African Openbill_aves_adult_ciconiiformes_ciconiidae_anastomus_lamelligerus",
"African Sacred Ibis_adult_pelecaniformes_threskiornithidae_threskiornis_aethiopicus": "African Sacred Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_aethiopicus",
"African Spoonbill_adult_pelecaniformes_threskiornithidae_platalea_alba": "African Spoonbill_aves_adult_pelecaniformes_threskiornithidae_platalea_alba",
"African Spoonbill_chick_pelecaniformes_threskiornithidae_platalea_alba": "African Spoonbill_aves_chick_pelecaniformes_threskiornithidae_platalea_alba",
"Aleutian Tern_adult_charadriiformes_laridae_onychoprion_aleuticus": "Aleutian Tern_aves_adult_charadriiformes_laridae_onychoprion_aleuticus",
"Australasian Swamphen_adult_gruiformes_rallidae_porphyrio_melanotus": "Australasian Swamphen_aves_adult_gruiformes_rallidae_porphyrio_melanotus",
"Australasian Swamphen_chick_gruiformes_rallidae_porphyrio_melanotus": "Australasian Swamphen_aves_chick_gruiformes_rallidae_porphyrio_melanotus",
"Australian Magpie_adult_passeriformes_artamidae_gymnorhina_tibicen": "Australian Magpie_aves_adult_passeriformes_artamidae_gymnorhina_tibicen",
"Australian Pelican_adult_pelecaniformes_pelecanidae_pelecanus_conspicillatus": "Australian Pelican_aves_adult_pelecaniformes_pelecanidae_pelecanus_conspicillatus",
"Australian White Ibis_adult_pelecaniformes_threskiornithidae_threskiornis_molucca": "Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca",
"Australian Wood Duck_adult_anseriformes_anatidae_chenonetta_jubata": "Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata",
"Bar-tailed Godwit_adult_charadriiformes_scolopacidae_limosa_lapponica": "Bar-tailed Godwit_aves_adult_charadriiformes_scolopacidae_limosa_lapponica",
"Bird_chick_unknown_unknown_unknown_unknown": "Bird_aves_chick_unknown_unknown_unknown_unknown",
"Bird_unknown_unknown_unknown_unknown_unknown": "Bird_aves_unknown_unknown_unknown_unknown_unknown",
"Black Swan_adult_anseriformes_anatidae_cygnus_atratus": "Black Swan_aves_adult_anseriformes_anatidae_cygnus_atratus",
"Caspian Tern_adult_charadriiformes_laridae_hydroprogne_caspia": "Caspian Tern_aves_adult_charadriiformes_laridae_hydroprogne_caspia",
"Cattle Egret_adult_pelecaniformes_ardeidae_bubulcus_ibis": "Cattle Egret_aves_adult_pelecaniformes_ardeidae_bubulcus_ibis",
"Chilean Flamingo_adult_phoenicopteriformes_phoenicopteridae_phoenicopterus_chilensis": "Chilean Flamingo_aves_adult_phoenicopteriformes_phoenicopteridae_phoenicopterus_chilensis",
"Chilean Flamingo_chick_phoenicopteriformes_phoenicopteridae_phoenicopterus_chilensis": "Chilean Flamingo_aves_chick_phoenicopteriformes_phoenicopteridae_phoenicopterus_chilensis",
"Chinstrap Penguin_adult_sphenisciformes_spheniscidae_pygoscelis_antarcticus": "Chinstrap Penguin_aves_adult_sphenisciformes_spheniscidae_pygoscelis_antarcticus",
"Chinstrap Penguin_chick_sphenisciformes_spheniscidae_pygoscelis_antarcticus": "Chinstrap Penguin_aves_chick_sphenisciformes_spheniscidae_pygoscelis_antarcticus",
"Common Tern_adult_charadriiformes_laridae_sterna_hirundo": "Common Tern_aves_adult_charadriiformes_laridae_sterna_hirundo",
"Domestic Duck_adult_anseriformes_anatidae_anas_platyrhynchos domesticus": "Domestic Duck_aves_adult_anseriformes_anatidae_anas_platyrhynchos domesticus",
"Dusky Moorhen_adult_gruiformes_rallidae_gallinula_tenebrosa": "Dusky Moorhen_aves_adult_gruiformes_rallidae_gallinula_tenebrosa",
"Eurasian Coot_adult_gruiformes_rallidae_fulica_atra": "Eurasian Coot_aves_adult_gruiformes_rallidae_fulica_atra",
"European Herring Gull_adult_charadriiformes_laridae_larus_argentatus": "European Herring Gull_aves_adult_charadriiformes_laridae_larus_argentatus",
"European Herring Gull_chick_charadriiformes_laridae_larus_argentatus": "European Herring Gull_aves_chick_charadriiformes_laridae_larus_argentatus",
"Gentoo Penguin_adult_sphenisciformes_spheniscidae_pygoscelis_papua": "Gentoo Penguin_aves_adult_sphenisciformes_spheniscidae_pygoscelis_papua",
"Great Knot_adult_charadriiformes_scolopacidae_calidris_tenuirostris": "Great Knot_aves_adult_charadriiformes_scolopacidae_calidris_tenuirostris",
"Grey Teal_adult_anseriformes_anatidae_anas_gracilis": "Grey Teal_aves_adult_anseriformes_anatidae_anas_gracilis",
"Gull-billed Tern_adult_charadriiformes_laridae_gelochelidon_nilotica": "Gull-billed Tern_aves_adult_charadriiformes_laridae_gelochelidon_nilotica",
"Hardhead_adult_anseriformes_anatidae_aythya_australis": "Hardhead_aves_adult_anseriformes_anatidae_aythya_australis",
"Lesser Black-backed Gull_adult_charadriiformes_laridae_larus_fuscus": "Lesser Black-backed Gull_aves_adult_charadriiformes_laridae_larus_fuscus",
"Lesser Black-backed Gull_chick_charadriiformes_laridae_larus_fuscus": "Lesser Black-backed Gull_aves_chick_charadriiformes_laridae_larus_fuscus",
"Little Grebe_adult_podicipediformes_podicipedidae_tachybaptus_ruficollis": "Little Grebe_aves_adult_podicipediformes_podicipedidae_tachybaptus_ruficollis",
"Magpie Lark_adult_passeriformes_monarchidae_grallina_cyanoleuca": "Magpie Lark_aves_adult_passeriformes_monarchidae_grallina_cyanoleuca",
"Masked Lapwing_adult_charadriiformes_charadriidae_vanellus_miles": "Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles",
"Mediterranean Gull_adult_charadriiformes_laridae_ichthyaetus_melanocephalus": "Mediterranean Gull_aves_adult_charadriiformes_laridae_ichthyaetus_melanocephalus",
"Mediterranean Gull_chick_charadriiformes_laridae_ichthyaetus_melanocephalus": "Mediterranean Gull_aves_chick_charadriiformes_laridae_ichthyaetus_melanocephalus",
"Muscovy Duck_adult_anseriformes_anatidae_cairina_moschata": "Muscovy Duck_aves_adult_anseriformes_anatidae_cairina_moschata",
"Pacific Black Duck_adult_anseriformes_anatidae_anas_superciliosa": "Pacific Black Duck_aves_adult_anseriformes_anatidae_anas_superciliosa",
"Pied Avocet_adult_charadriiformes_recurvirostridae_recurvirostra_avosetta": "Pied Avocet_aves_adult_charadriiformes_recurvirostridae_recurvirostra_avosetta",
"Pied Butcherbird_adult_passeriformes_artamidae_cracticus_nigrogularis": "Pied Butcherbird_aves_adult_passeriformes_artamidae_cracticus_nigrogularis",
"Pied Oystercatcher_adult_charadriiformes_haematopodidae_haematopus_longirostris": "Pied Oystercatcher_aves_adult_charadriiformes_haematopodidae_haematopus_longirostris",
"Pied Stilt_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus": "Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus",
"Reed Cormorant_adult_suliformes_phalacrocoracidae_microcarbo_africanus": "Reed Cormorant_aves_adult_suliformes_phalacrocoracidae_microcarbo_africanus",
"Royal Spoonbill_adult_pelecaniformes_threskiornithidae_platalea_regia": "Royal Spoonbill_aves_adult_pelecaniformes_threskiornithidae_platalea_regia",
"Silver Gull_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae": "Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae",
"Straw-necked Ibis_adult_pelecaniformes_threskiornithidae_threskiornis_spinicollis": "Straw-necked Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_spinicollis",
"Torresian Crow_adult_passeriformes_corvidae_corvus_orru": "Torresian Crow_aves_adult_passeriformes_corvidae_corvus_orru",
"Welcome Swallow_adult_passeriformes_hirundinidae_hirundo_neoxena": "Welcome Swallow_aves_adult_passeriformes_hirundinidae_hirundo_neoxena",
"Whimbrel_adult_charadriiformes_scolopacidae_numenius_phaeopus": "Whimbrel_aves_adult_charadriiformes_scolopacidae_numenius_phaeopus",
"White-faced Heron_adult_pelecaniformes_ardeidae_egretta_novaehollandiae": "White-faced Heron_aves_adult_pelecaniformes_ardeidae_egretta_novaehollandiae",
"Yellow-billed Stork_adult_ciconiiformes_ciconiidae_mycteria_ibis": "Yellow-billed Stork_aves_adult_ciconiiformes_ciconiidae_mycteria_ibis",
"Yellow-billed Stork_chick_ciconiiformes_ciconiidae_mycteria_ibis": "Yellow-billed Stork_aves_chick_ciconiiformes_ciconiidae_mycteria_ibis"
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
        # iterate through files in dir
        for key, value in com_sci.items():
            for root, _, files in os.walk(file_path_variable):
                for file in files:
                    if file.endswith(".json"):
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r') as f:
                            filedata = f.read()
                            filedata = filedata.replace(key, value)
                            print(key, value)
                        with open(filepath, 'w') as f:
                            f.write(filedata)