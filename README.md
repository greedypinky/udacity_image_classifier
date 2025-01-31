# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


## Develop code for an image classifier built with PyTorch
File is defined in the `Image Classifier Project.ipynb` file

##  Command line application to train the model with training and test data and predict a flower image that is unseen by the model.

### How to train the model
File is defined under `train.py`
```
python train.py "flowers" --save_dir "save_model" --arch "vgg16" --learning_rate 0.001 --hidden_units 512 --epochs 1
```

### After train the model, we can predit the model and plot the top class K probability in a graph.
File is defined under `predict.py`
```
python predict.py 'flowers/train/1/image_06734.jpg' 'checkpoint.pth' --top_k 5 --category_names 'cat_to_name.json' --gpu
```

### training and test data is under `flower` folder

