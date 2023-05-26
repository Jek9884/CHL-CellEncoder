# CHL-CellEncoder

Project for Computational Health Laboratory exam. \
Authors:
\
Martina Melero Cavallo \
Francesco Mitola \
Alessandro Capurso 

## Project track
Train an encoder for single cell data, using a transformer architecture.
The encoder has to yield an embedding for each cell.
Use Tabula muris data [1], augmented by including missing reads for a small number of random genes. Compare the results obtained during clustering of cells (cell identification) with and without using the embedding. Compare also with embeddings from ref [2]

References:
[1]https://tabula-muris.ds.czbiohub.org/ 
[2]https://docs.scvi-tools.org/en/stable/user_guide/models/scvi.html 

## Execution
There are different .py files that are scripts to perform specific tasks (finetuning, embedding generation, etc.).
The notebooks are employed to visualize clustering results and compare the different approaches.
