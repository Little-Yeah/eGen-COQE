# Comparative Opinion Quintuple Extraction (COQE)

This repo contains the annotated data and code for our paper [eGen: An Enhanced Generative Framework for Comparative Opinion Quintuple Extraction]


## Short Summary 
- We aim to tackle COQE task: given a sentence, we predict all comparative quads `subject (\textit{sub}), object (\textit{obj}), comparative aspect (\textit{ca}), comparative opinion (\textit{co}), and comparative preference (\textit{cp})`

## Data
- We use two three datasets, Camera-COQE, Car-COQE and Ele-COQE:

- **Camera-COQE**: On basis of the Camera domain corpus released by Kessler and Kuhn (2014), we completed the annotation of Comparative Opinion and Comparative Preference for 1705 comparative sentences, and introducing 1599 non-comparative sentences.
  
- **Car-COQE**: Based on the COAE 2012/2013 Car domain corpus, we supplemented with the annotation of Comparative Opinion and Comparative Preference.
  
- **Ele-COQE**: Similar to Car-COQE, we construct the Ele-COQE dataset based on the COAE 2012/2013 Electronics (Ele) domain corpus.


## Requirements

We highly recommend you to install the specified version of the following packages to avoid unnecessary troubles:
- transformers==4.0.0
- sentencepiece==0.1.91
- pytorch_lightning==0.8.1


## Quick Start

- Set up the environment as described in the above section
- Download the pre-trained T5-base model (you can also use larger versions for better performance depending on the availability of the computation resource), put it under the folder `T5-base`.
  - You can also skip this step and the pre-trained model would be automatically downloaded to the cache in the next step
- Run command `sh run.sh`.
- More details can be found in the paper and the help info in the `main.py`.


## Citation

If the code is used in your research, please star our repo and cite our paper as follows:

```
