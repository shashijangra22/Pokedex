# Pokedex
A CNN based Pokemon Classifier.

## Steps to train the network
`python3 train.py -d dataset/ -m pokedex.model -l lb.pickle`

## Steps to classify the network
`python3 classify.py -i examples/pikachu.jpg -m pokedex.model -l lb.pickle`
