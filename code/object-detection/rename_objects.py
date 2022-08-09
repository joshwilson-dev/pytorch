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
"littleblackcormorant-aves-suliformes-phalacrocoracidae-phalacrocorax-sulcirostris-adult-unknown": " Little Black Cormorant_aves_suliformes_phalacrocoracidae_phalacrocorax_sulcirostris_adult",
"africandarter-aves-suliformes-anhingidae-anhinga-rufa-adult-unknown": "African Darter_aves_suliformes_anhingidae_anhinga_rufa_adult",
"africandarter-aves-suliformes-anhingidae-anhinga-rufa-juvenile-unknown": "African Darter_aves_suliformes_anhingidae_anhinga_rufa_chick",
"africanopenbill-aves-ciconiiformes-ciconiidae-anastomus-lamelligerus-adult-unknown": "African Openbill_aves_ciconiiformes_ciconiidae_anastomus_lamelligerus_adult",
"africansacredibis-aves-pelecaniformes-threskiornithidae-threskiornis-aethiopicus-adult-unknown": "African Sacred Ibis_aves_pelecaniformes_threskiornithidae_threskiornis_aethiopicus_adult",
"africanspoonbill-aves-pelecaniformes-threskiornithidae-platalea-alba-adult-unknown": "African Spoonbill_aves_pelecaniformes_threskiornithidae_platalea_alba_adult",
"africanspoonbill-aves-pelecaniformes-threskiornithidae-platalea-alba-chick-unknown": "African Spoonbill_aves_pelecaniformes_threskiornithidae_platalea_alba_chick",
"aleutiantern-aves-charadriiformes-laridae-onychoprion-aleuticus-adult-unknown": "Aleutian Tern_aves_charadriiformes_laridae_onychoprion_aleuticus_adult",
"australasianswamphen-aves-gruiformes-rallidae-porphyrio-melanotus-adult-unknown": "Australasian Swamphen_aves_gruiformes_rallidae_porphyrio_melanotus_adult",
"australasianswamphen-aves-gruiformes-rallidae-porphyrio-melanotus-juvenile-unknown": "Australasian Swamphen_aves_gruiformes_rallidae_porphyrio_melanotus_chick",
"australianmagpie-aves-passeriformes-artamidae-gymnorhina-tibicen-adult-unknown": "Australian Magpie_aves_passeriformes_artamidae_gymnorhina_tibicen_adult",
"australianpelican-aves-pelecaniformes-pelecanidae-pelecanus-conspicillatus-adult-unknown": "Australian Pelican_aves_pelecaniformes_pelecanidae_pelecanus_conspicillatus_adult",
"australianwhiteibis-aves-pelecaniformes-threskiornithidae-threskiornis-molucca-adult-unknown": "Australian White Ibis_aves_pelecaniformes_threskiornithidae_threskiornis_molucca_adult",
"australianwoodduck-aves-anseriformes-anatidae-chenonetta-jubata-adult-unknown": "Australian Wood Duck_aves_anseriformes_anatidae_chenonetta_jubata_adult",
"bartailedgodwit-aves-charadriiformes-scolopacidae-limosa-lapponica-adult-unknown": "Bar-tailed Godwit_aves_charadriiformes_scolopacidae_limosa_lapponica_adult",
"bird-aves-unknown-unknown-unknown-unknown-unknown-chick": "Bird_aves_unknown_unknown_unknown_unknown_chick",
"bird-aves-unknown-unknown-unknown-unknown-chick-unknown": "Bird_aves_unknown_unknown_unknown_unknown_chick",
"bird-aves-unknown-unknown-unknown-unknown-unknown-unknown": "Bird_aves_unknown_unknown_unknown_unknown_unknown",
"blackswan-aves-anseriformes-anatidae-cygnus-atratus-adult-unknown": "Black Swan_aves_anseriformes_anatidae_cygnus_atratus_adult",
"caspiantern-aves-charadriiformes-laridae-hydroprogne-caspia-adult-unknown": "Caspian Tern_aves_charadriiformes_laridae_hydroprogne_caspia_adult",
"cattleegret-aves-pelecaniformes-ardeidae-bubulcus-ibis-adult-unknown": "Cattle Egret_aves_pelecaniformes_ardeidae_bubulcus_ibis_adult",
"chileanflamingo-aves-phoenicopteriformes-phoenicopteridae-phoenicopterus-chilensis-adult-unknown": "Chilean Flamingo_aves_phoenicopteriformes_phoenicopteridae_phoenicopterus_chilensis_adult",
"chileanflamingo-aves-phoenicopteriformes-phoenicopteridae-phoenicopterus-chilensis-juvenile-unknown": "Chilean Flamingo_aves_phoenicopteriformes_phoenicopteridae_phoenicopterus_chilensis_chick",
"chinstrappenguin-aves-sphenisciformes-spheniscidae-pygoscelis-antarcticus-adult-unknown": "Chinstrap Penguin_aves_sphenisciformes_spheniscidae_pygoscelis_antarcticus_adult",
"chinstrappenguin-aves-sphenisciformes-spheniscidae-pygoscelis-antarcticus-juvenile-unknown": "Chinstrap Penguin_aves_sphenisciformes_spheniscidae_pygoscelis_antarcticus_chick",
"commontern-aves-charadriiformes-laridae-sterna-hirundo-adult-unknown": "Common Tern_aves_charadriiformes_laridae_sterna_hirundo_adult",
"domesticduck-aves-anseriformes-anatidae-anas-platyrhynchosdomesticus-adult-unknown": "Domestic Duck_aves_anseriformes_anatidae_anas_platyrhynchosdomesticus_adult",
"duskymoorhen-aves-gruiformes-rallidae-gallinula-tenebrosa-adult-unknown": "Dusky Moorhen_aves_gruiformes_rallidae_gallinula_tenebrosa_adult",
"eurasiancoot-aves-gruiformes-rallidae-fulica-atra-adult-unknown": "Eurasian Coot_aves_gruiformes_rallidae_fulica_atra_adult",
"europeanherringgull-aves-charadriiformes-laridae-larus-argentatus-adult-unknown": "European Herring Gull_aves_charadriiformes_laridae_larus_argentatus_adult",
"europeanherringgull-aves-charadriiformes-laridae-larus-argentatus-juvenile-unknown": "European Herring Gull_aves_charadriiformes_laridae_larus_argentatus_chick",
"gentoopenguin-aves-sphenisciformes-spheniscidae-pygoscelis-papua-adult-unknown": "Gentoo Penguin_aves_sphenisciformes_spheniscidae_pygoscelis_papua_adult",
"greatknot-aves-charadriiformes-scolopacidae-calidris-tenuirostris-adult-unknown": "Great Knot_aves_charadriiformes_scolopacidae_calidris_tenuirostris_adult",
"greyteal-aves-anseriformes-anatidae-anas-gracilis-adult-unknown": "Grey Teal_aves_anseriformes_anatidae_anas_gracilis_adult",
"gullbilledtern-aves-charadriiformes-laridae-gelochelidon-nilotica-adult-unknown": "Gull-billed Tern_aves_charadriiformes_laridae_gelochelidon_nilotica_adult",
"hardhead-aves-anseriformes-anatidae-aythya-australis-adult-unknown": "Hardhead_aves_anseriformes_anatidae_aythya_australis_adult",
"lesserblackbackedgull-aves-charadriiformes-laridae-larus-fuscus-adult-unknown": "Lesser Black-backed Gull_aves_charadriiformes_laridae_larus_fuscus_adult",
"lesserblackbackedgull-aves-charadriiformes-laridae-larus-fuscus-juvenile-unknown": "Lesser Black-backed Gull_aves_charadriiformes_laridae_larus_fuscus_chick",
"littlegrebe-aves-podicipediformes-podicipedidae-tachybaptus-ruficollis-adult-unknown": "Little Grebe_aves_podicipediformes_podicipedidae_tachybaptus_ruficollis_adult",
"magpielark-aves-passeriformes-monarchidae-grallina-cyanoleuca-adult-unknown": "Magpie Lark_aves_passeriformes_monarchidae_grallina_cyanoleuca_adult",
"maskedlapwing-aves-charadriiformes-charadriidae-vanellus-miles-adult-unknown": "Masked Lapwing_aves_charadriiformes_charadriidae_vanellus_miles_adult",
"mediterraneangull-aves-charadriiformes-laridae-ichthyaetus-melanocephalus-adult-unknown": "Mediterranean Gull_aves_charadriiformes_laridae_ichthyaetus_melanocephalus_adult",
"mediterraneangull-aves-charadriiformes-laridae-ichthyaetus-melanocephalus-juvenile-unknown": "Mediterranean Gull_aves_charadriiformes_laridae_ichthyaetus_melanocephalus_chick",
"muscovyduck-aves-anseriformes-anatidae-cairina-moschata-adult-unknown": "Muscovy Duck_aves_anseriformes_anatidae_cairina_moschata_adult",
"muscovyduck-aves-anseriformes-anatidae-cairina-moschata-adult-male": "Muscovy Duck_aves_anseriformes_anatidae_cairina_moschata_adult",
"pacificblackduck-aves-anseriformes-anatidae-anas-superciliosa-adult-unknown": "Pacific Black Duck_aves_anseriformes_anatidae_anas_superciliosa_adult",
"piedavocet-aves-charadriiformes-recurvirostridae-recurvirostra-avosetta-adult-unknown": "Pied Avocet_aves_charadriiformes_recurvirostridae_recurvirostra_avosetta_adult",
"piedbutcherbird-aves-passeriformes-artamidae-cracticus-nigrogularis-adult-unknown": "Pied Butcherbird_aves_passeriformes_artamidae_cracticus_nigrogularis_adult",
"piedoystercatcher-aves-charadriiformes-haematopodidae-haematopus-longirostris-adult-unknown": "Pied Oystercatcher_aves_charadriiformes_haematopodidae_haematopus_longirostris_adult",
"piedstilt-aves-charadriiformes-recurvirostridae-himantopus-leucocephalus-adult-unknown": "Pied Stilt_aves_charadriiformes_recurvirostridae_himantopus_leucocephalus_adult",
"reedcormorant-aves-suliformes-phalacrocoracidae-microcarbo-africanus-adult-unknown": "Reed Cormorant_aves_suliformes_phalacrocoracidae_microcarbo_africanus_adult",
"royalspoonbill-aves-pelecaniformes-threskiornithidae-platalea-regia-adult-unknown": "Royal Spoonbill_aves_pelecaniformes_threskiornithidae_platalea_regia_adult",
"silvergull-aves-charadriiformes-laridae-chroicocephalus-novaehollandiae-adult-unknown": "Silver Gull_aves_charadriiformes_laridae_chroicocephalus_novaehollandiae_adult",
"strawneckedibis-aves-pelecaniformes-threskiornithidae-threskiornis-spinicollis-adult-unknown": "Straw-necked Ibis_aves_pelecaniformes_threskiornithidae_threskiornis_spinicollis_adult",
"torresiancrow-aves-passeriformes-corvidae-corvus-orru-adult-unknown": "Torresian Crow_aves_passeriformes_corvidae_corvus_orru_adult",
"welcomeswallow-aves-passeriformes-hirundinidae-hirundo-neoxena-adult-unknown": "Welcome Swallow_aves_passeriformes_hirundinidae_hirundo_neoxena_adult",
"whimbrel-aves-charadriiformes-scolopacidae-numenius-phaeopus-adult-unknown-aves-charadriiformes-scolopacidae-numenius-phaeopus-adult-unknown": "Whimbrel_aves_charadriiformes_scolopacidae_numenius_phaeopus_adult",
"whitefacedheron-aves-pelecaniformes-ardeidae-egretta-novaehollandiae-adult-unknown": "White-faced Heron_aves_pelecaniformes_ardeidae_egretta_novaehollandiae_adult",
"yellowbilledstork-aves-ciconiiformes-ciconiidae-mycteria-ibis-adult-unknown": "Yellow-billed Stork_aves_ciconiiformes_ciconiidae_mycteria_ibis_adult",
"yellowbilledstork-aves-ciconiiformes-ciconiidae-mycteria-ibis-juvenile-unknown": "Yellow-billed Stork_aves_ciconiiformes_ciconiidae_mycteria_ibis_chick"
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