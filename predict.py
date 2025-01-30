import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import json
from torchvision import datasets, transforms, models
from collections import OrderedDict
from get_input_args import get_input_args_predict,  check_predict_command_line_arguments


alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models_options = {'alexnet': alexnet, 'vgg16': vgg16, }

# helper methods


def get_model(model_name='vgg16'):
    try:
        model = models_options[model_name]
        return model

    except ValueError as e:
        print(f"Error: {e}")
        return None


def map_cat_label_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def save_the_checkpoint(model, checkpointfile, image_datasets, epoch, optimizer, classifier):
    ''' save the trained model state as checkpoint
    '''
    model.class_to_idx = image_datasets.class_to_idx
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classifier': classifier,
        'class_to_idx': model.class_to_idx
    }
    # save model checkpoint as for eg. checkpoint.pth
    torch.save(checkpoint, checkpointfile)


def load_checkpoint_with_file(checkpointfile):
    ''' helper function to load the trained model
        returns model
    '''
    try:
        checkpointpath = os.path.join("save_model", checkpointfile)
        if os.path.exists(checkpointpath):
            print(f"The file '{checkpointpath}' exists.")
        else:
            print(f"Error: No such file or directory: '{checkpointpath}'")

        # checkpoint = torch.load('checkpoint.pth')
        checkpoint = torch.load(checkpointpath)

        print(f"what is the model name? {checkpoint['model_name']}")
        model = get_model(checkpoint['model_name'])
        if model:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.classifier = checkpoint['classifier']
            model.class_to_index = checkpoint['class_to_idx']

            for param in model.parameters():
                param.requires_grad = False
            print("model is loaded")
            return model
        else:
            raise ValueError("No model is loaded from in_arg.checkpoint")
    except FileNotFoundError as e:
        print(f"Load checkpoint from checkpoint file exception: {e}")


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # The output of the convolutional layers (before the fully connected layer)
    #  has a shape of (batch_size, 512, 7, 7)
    # Open the image
    with Image.open(image) as im:
        # resize, crop to 224, transfer to tensor and normalize
        transform_image = transforms.Compose([
            # transforms.Resize(256),
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        pil_image = transform_image(im)
        print(f"PIL image shape: {pil_image.shape}")
        return pil_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # eg. image_path = 'flowers/train/1/image_06734.jpg'
    print(f"debug what is the image path? {image_path}")
    img = process_image(image_path)
    print("debug: what is the shape of the processed image?")
    img = img.unsqueeze(0)
    print(img.shape)

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    with torch.no_grad():
        output = model(img)

    # from logits to probabilities
    ps = torch.softmax(output, dim=1)
    # ps = torch.exp(output)

    # get the topK probablities and index
    # [batchsize, top k of probability], [batchsize, top k of index]
    top_pros, top_indices = ps.topk(topk)

    print(
        f"debug do we really get the value topPros, topIndex? {top_pros},{top_indices}")
    # convert from these indices to the actual class labels
    # using class_to_idx
    top_pros = top_pros.numpy().squeeze()  # [0.3, 0.4, 0.1, 0.1, 0.1]
    top_indices = top_indices.numpy().squeeze()
    # convert indices to class
    print("debug what is the class index:")
    print(type(model.class_to_index))
    print(model.class_to_index)
    idx_to_class_map = {idx: cls for cls, idx in model.class_to_index.items()}
    # get the top 5 class
    top_class = [idx_to_class_map[idx] for idx in top_indices]
    print(f"can I get the top class? {top_class}")
    return top_pros, top_class


def show_prob_with_image(img, top_ps, top_class, top_k):
    ''' Display an image along with the top K classes
    '''
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), nrows=2)
    img = img.numpy().transpose((1, 2, 0))
    print(f"after transpose {img.shape}")

    ax1.imshow(np.clip(img, 0, 1))
    ax1.axis('off')
    # ax2.barh(np.arange(5), top5_ps)
    ax2.barh(np.arange(top_k), top_ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(top_k))
    cat_to_name = map_cat_label_to_name()
    flower_names = [cat_to_name.get(cls, "Unknown") for cls in top_class]
    # debug
    print("can we get the flower names?")
    print(flower_names)
    ax2.set_yticklabels(flower_names)
    # TOOD: title need to set the max probabilty flower name!!
    print("what is max ps index?")
    # top_ps 'numpy.ndarray', no index method
    top_ps = top_ps.tolist()
    top_index = top_ps.index(max(top_ps))
    title = cat_to_name[str(top_class[top_index])]
    print(title)
    ax1.set_title(title)
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()


def main():
    ''' add prediction
    '''
    # do some validation for arguments
    in_arg = get_input_args_predict()
    # Function that checks command line arguments using in_arg
    check_predict_command_line_arguments(in_arg)

    print("loading model...")
    model = load_checkpoint_with_file(in_arg.checkpoint)
    if model:
        # image_path = 'flowers/train/1/image_06734.jpg'
        image_path = in_arg.image_dir
        topk = int(in_arg.top_k) if in_arg.top_k else 5
        top5_ps, top5_class = predict(image_path, model, topk)
        print(
            f"debug do we really get the value top5_ps,top5_class? {top5_ps},{top5_class}")
        show_prob_with_image(process_image(image_path),
                             top5_ps, top5_class, topk)
    else:
        print("No model is loaded from in_arg.checkpoint and predict cannot be proceeded!")


# Call to main function to run the program
if __name__ == '__main__':
    main()
