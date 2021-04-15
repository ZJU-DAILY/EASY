# EASY

Source code for Make it Easy: An Effective End-to-End Entity Alignment Framework. SIGIR 2021.

<p align="center">
    <br>
    <img alt="Overview" src="https://raw.githubusercontent.com/underreview/EASY/main/figure/overview.png" width="100%"/>
    <br>
<p>

## Installation

To run our code, first install required packages. Then run preprocess

    pip install -r requirements.txt
    sh preprocess.sh

## Run 

Run on all dataset with default settings

First get NEAP results.

    sh neap.sh

Then get SRS results.

    python main.py --pair all    

## Run on specific dataset/settings

### NEAP

The SRS process need the result of NEAP. To get NEAP results on a specific dataset(e.g. en_fr)

    python neap.py --pair en_fr
    
    
For fasttext, please download [aligned word vectors](https://fasttext.cc/docs/en/aligned-vectors.html) <code>wiki.{lang}.align.vec</code> and place them
into <code>aligned_vectors/</code> folder.

    
    mkdir aligned_vectors
    cd aligned_vectors
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.fr.align.vec
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.de.align.vec


### SRS

After acquiring similarity matrices from NEAP, run main.py to refine. 

    python main.py --pair en_fr 

Change arguments for different settings. To get help on arugments, run 

    python main.py --help

The refinement process is based on the code of [MRAEA](https://github.com/MaoXinn/MRAEA),
[RREA](https://github.com/MaoXinn/RREA), 
[GCN-Align](https://github.com/1049451037/GCN-Align).  In our experiment, training is done on CPU.


## Acknowledgement

We use the code of 
[MRAEA](https://github.com/MaoXinn/MRAEA),
[RREA](https://github.com/MaoXinn/RREA), 
[GCN-Align](https://github.com/1049451037/GCN-Align),
[DGMC](https://github.com/rusty1s/deep-graph-matching-consensus),
[AttrGNN](https://github.com/thunlp/explore-and-evaluate),
[OpenEA](https://github.com/nju-websoft/OpenEA),
[EAKit](https://github.com/THU-KEG/EAKit),
[SimAlign](https://github.com/cisnlp/simalign). 

DBP15k dataset is from [GMNN](https://github.com/syxu828/Crosslingula-KG-Matching) and 
[AttrGNN](https://github.com/thunlp/explore-and-evaluate).

SRPRS dataset is from [RSN](https://github.com/nju-websoft/RSN). 
