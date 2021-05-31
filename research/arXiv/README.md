# arXiv classifier

This is a project to scrape arXiv papers and classify them using a state of the art classifiers **BERT** and **SciBERT**. 

Scraper used for this project is present in https://github.com/cellcomplexitylab/gufi . Feel free to use it and create issues for me 
if it doesn't perform as per your need or for any customization.

## Raw and preprocessed Data

All the raw and preprocessed data is present in **GoogleDrive**
 feel free to ask [Professor Guillaume](https://github.com/gui11aume) for its access.

## exp-01 (classifying using the Title only)

This was an attempt to just use the Title of the papers and attempt to classify them using different transformer models.

### exp-01-01 (Using BERT with its pre-trained tokenizer)

the notebook `BERT_exp_01_01.ipynb` implements the model using **PyTorch** and **HuggingFace**, the model is build using PyTorch Lightning because it make the GPU usage and parallelization easier.

### exp-02-01 (Using SciBERT with its pre-trained tokenizer)

the notebook `Sci_BERT_exp_02_01.ipynb` implements the model using **PyTorch** and **HuggingFace**, the model is build using PyTorch Lightning because it make the GPU usage and parallelization easier.

## Results and models

All the models are again present in **GoogleDrive**
abd the results are summarized in the `arXiv_Classifier.pdf`