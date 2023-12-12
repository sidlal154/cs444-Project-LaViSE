# cs444-Project-LaViSE
CS444 project: LaViSE with pluggable embeddings

This project code is heavily based on the reproduciability code referen here: https://github.com/ErikBuis/FACT2023, which is inturn refered from the original paper's code here: https://github.com/YuYang0901/LaViSE. All the changes added (additional models, embeddings, etc) have been marked with a comment 'CS444'.

We have implemented two sets of code which are highly similar but only differ by the embeddings used.

For GloVE and FastText (Falavise): use the cs444-project_falavise directory and cs444_project_falavise.ipyn ntoebook. Do take case to switch between fasttext and glove while preprocessing in the notebook, and while running the training and inference scripts by changing the --word-model flag. 

For BERT (BeLavise): use the cs444-project_falavise directory and cs444_project_falavise.ipyn ntoebook. Note that there is no '--word-model' flag here.

This project is desgined and developed by Siddharth Lal (sl203)
