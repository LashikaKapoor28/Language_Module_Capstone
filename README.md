# Language_Module_Capstone

## Added basic project file system:
coco.py (Apurva) - Coco JSON parser and ID mapping  
embedder.py (Lashika) - Text tokenization and the glove vectorizer  
model.py (Shanya and Evgeny) - The actual model (pytorch) + save/load  
train.py (Sriram) - Training loop + gradient descent steps  
database.py (Nihanth) - Matrix search on DB and image loader  
main.py (Whoever done) -  Actual runnable final script  

Trained weights are in data/W_embed.npy (512 to 200). Re-run train.py to regenerate

Libraries:
numpy
mygrad
gensim
requests
pillow
torch
torchvision