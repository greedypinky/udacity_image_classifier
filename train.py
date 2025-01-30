# Check torch version and CUDA status if GPU is enabled.
# Train a new network on a data set with train.py
# * Basic usage: python train.py data_directory
# * Prints out training loss, validation loss, and validation accuracy as the network trains
# * Options: * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory *
# Choose architecture: python train.py data_dir --arch "vgg13" * Set hyperparameters: python train.py
# data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 * Use GPU for training: python train.py
# data_dir --gpu
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import json
from torchvision import datasets, transforms, models
from collections import OrderedDict
from get_input_args import get_input_args_train
from get_input_args import check_training_command_line_arguments, check_predict_command_line_arguments

alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
# alexnet = models.alexnet(AlexNet_Weights.DEFAULT)
# vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)

models_options = {'alexnet': alexnet, 'vgg16': vgg16, }

# Helper methods


def load_checkpoint_with_file(checkpointfile):
    ''' helper function to load the trained model
        returns model
    '''
    checkpoint = torch.load(checkpointfile)
    try:
        model = get_model(checkpoint['model_name'])
        print(f"what is the arch? {checkpoint['model_name']}")
        if model:
            model.classifier = checkpoint['classifier']
            model.load_state_dict(checkpoint['model_state_dict'])
            model.class_to_index = checkpoint['class_to_idx']

            for param in model.parameters():
                param.requires_grad = False
        else:
            raise ValueError("No model is loaded from in_arg.checkpoint")

    except Exception as e:
        print(f"Load checkpoint from checkpoint file exception: {e}")
    return model


def get_model(model_name):
    try:
        model = models_options[model_name]
        return model

    except ValueError as e:
        print(f"Error: {e}")
        return None


def init_model_classifier(model_name, model, hidden_units=512):
    '''
        create classifier per model type, use the hidden_units provided, otherwise will use 512 as default.
    '''
    # need to freeze the layer first before modify the classifier
    for param in model.parameters():
        param.requires_grad = False
    # need to train the classifer for our data

    print(f"what is hidden layer? {hidden_units}")
    if model_name == 'vgg16':
        # TODO: verify the hidden units needs to be within some ranges, cannot be larger than the features or less than 102
        if hidden_units:
            # suggested by mentor but it doesn't improve much
            # classifier = nn.Sequential(OrderedDict(
            #     [('fc1', nn.Linear(25088, 256)),
            #      ('relu', nn.ReLU()),
            #      ('dropout', nn.Dropout(0.5)),
            #      ('fc2', nn.Linear(256, 102)),
            #      ('output', nn.LogSoftmax(dim=1))
            #      ]))
            # classifier = nn.Sequential(OrderedDict(
            #     [('fc1', nn.Linear(25088, 4096)),
            #      ('relu', nn.ReLU()),
            #         ('dropout', nn.Dropout(0.5)),
            #         # set the hidden layer here
            #         ('fc2', nn.Linear(4096, hidden_units)),
            #         ('relu', nn.ReLU()),
            #         ('dropout', nn.Dropout(0.5)),
            #         ('fc3', nn.Linear(hidden_units, 102)),
            #         ('output', nn.LogSoftmax(dim=1))
            #      ]))
            classifier = nn.Sequential(OrderedDict(
                [('fc1', nn.Linear(25088, hidden_units)),
                 ('relu', nn.ReLU()),
                 # regularize the model by randomly dropping 10% of neurons during training.
                 ('dropout', nn.Dropout(0.1)),
                 ('fc2', nn.Linear(hidden_units, 102)),
                 # ('output', nn.LogSoftmax(dim=1))
                 ]))
        else:
            default_hidden_units = 1000
            classifier = nn.Sequential(OrderedDict(
                [('fc1', nn.Linear(25088, default_hidden_units)),
                 ('relu', nn.ReLU()),
                 ('dropout', nn.Dropout(0.1)),
                 ('fc2', nn.Linear(default_hidden_units, 102)),
                 ('output', nn.LogSoftmax(dim=1))
                 ]))
    else:
        classifier = nn.Sequential(OrderedDict(
            [('fc1', nn.Linear(9216, 4096)),
             ('relu', nn.ReLU()),
             ('dropout', nn.Dropout(0.3)),
                # set the hidden layer here
                ('fc2', nn.Linear(4096, 2048)),
             ('relu', nn.ReLU()),
             ('dropout', nn.Dropout(0.3)),
             ('fc3', nn.Linear(2048, 102)),
             ('output', nn.LogSoftmax(dim=1))
             ]))

    model.classifier = classifier
    return model


def save_the_checkpoint(model, model_name, save_dir, image_datasets, epoch, optimizer, classifier):
    ''' save the trained model state as checkpoint
    '''
    model.class_to_idx = image_datasets.class_to_idx
    checkpoint = {
        'model_name': model_name,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classifier': classifier,
        'class_to_idx': model.class_to_idx
    }
    # save model checkpoint as for eg. checkpoint.pth
    torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))

# Train the model


def train(model, dataloader, testloader, learning_rate=0.001, epochs=5, device_mode="cpu"):
    # define the loss
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    # learning rate need to read from Argument
    learning_rate = learning_rate

    # define the optomizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # set device to cpu or cuda
    device = torch.device("cuda" if device_mode == "gpu" else "cpu")
    print(device)
    model.to(device)

    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 40
    for epoch in range(epochs):
        for images, labels in dataloader:
            steps += 1

            # move input, label tensores to the device
            images, labels = images.to(device), labels.to(device)
            # reset the gradient
            optimizer.zero_grad()
            # calculate the log probability
            # we should get 10 class probabilities for 64 examples
            logps = model.forward(images)

            # calculate the loss
            loss = criterion(logps, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print("new loss...")
            print(loss.item())

            # in every 5 loops, we will use the test data to check the model's accuracy
            if steps % print_every == 0:
                # turn off gradients
                with torch.no_grad():
                    # set the model to eval mode
                    print("set to eval mode")
                    model.eval()
                    # validation
                    test_loss, accuracy = test(
                        model, testloader, criterion, device)
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/len(dataloader):.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy%: {accuracy/len(testloader)*100:.3f}")

                # set the modal back to training mode
                model.train()
                print("reset to train mode")
    print("Training is done!")
    return optimizer


def test(model, testloader, criterion, device):
    """
        use testdata to test the model
    """
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        logps = model.forward(images)
        loss = criterion(logps, labels)
        test_loss += loss.item()

        # get the probability
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        # equality = (labels.data == ps.max(dim=1)[1])
        # accuracy += equality.type(torch.FloatTensor).mean()

        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        return test_loss, accuracy


def validation(model, validloader):
    """
    """
    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to('cpu'), labels.to('cpu')
            # Calculating the accuracy - take exponential to get the probabilities
            output = model(images)
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            # accuracy += equality.type_as(torch.FloatTensor()).mean()
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return accuracy


def main():
    ''' 
        1. load data for training
        2. train the model and test with eval mode.
        3. validate the model with images that are not seen by the model before.
        4. save the model with checkpoint metadata.
    '''
    # do some validation for arguments
    in_arg = get_input_args_train()
    # Function that checks command line arguments using in_arg
    check_training_command_line_arguments(in_arg)

    # data_dir = 'flowers' in our folder
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_image_datasets = datasets.ImageFolder(
        test_dir, transform=test_transforms)
    validation_image_datasets = datasets.ImageFolder(
        valid_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloader = torch.utils.data.DataLoader(
        image_datasets, batch_size=64, shuffle=True)
    # testloader = torch.utils.data.DataLoader(
    #     test_image_datasets, batch_size=64, drop_last=True)
    # validloader = torch.utils.data.DataLoader(
    #     validation_image_datasets, batch_size=64, drop_last=True)

    testloader = torch.utils.data.DataLoader(
        test_image_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(
        validation_image_datasets, batch_size=64, shuffle=True)

    model = get_model(in_arg.arch)
    if model:
        print("Model created successfully!")
    else:
        print("Failed to create model.")
    model = init_model_classifier(
        in_arg.arch, model, hidden_units=in_arg.hidden_units)

    # train(model, dataloader, testloader, learning_rate=in_arg.learning_rate,
    #       epochs=in_arg.epochs, device_mode=in_arg.gpu)

    save_the_optimizer = train(model, dataloader, validloader, learning_rate=in_arg.learning_rate,
                               epochs=in_arg.epochs, device_mode=in_arg.gpu)

    accuracy = validation(model, testloader)

    print("what is the accuracy?")
    print(accuracy)
    # save model to checkpoint
    print(save_the_optimizer)
    save_the_checkpoint(model, in_arg.arch, in_arg.save_dir, image_datasets,
                        in_arg.epochs, save_the_optimizer, model.classifier)

    checkpointfile = os.path.join(in_arg.save_dir, "checkpoint.pth")
    print(checkpointfile)
    saved_model = load_checkpoint_with_file(checkpointfile)
    print("after model is loaded!")

# Call to main function to run the program
if __name__ == '__main__':
    main()
