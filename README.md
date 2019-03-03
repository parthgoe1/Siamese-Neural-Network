# Face recognition using Siamese-Neural-Network (Pytorch)

Pytorch implementation of Siamese-Neural-Network. Finds similarity in faces using Siamese-Neural-Network. Siamese-Neural-Network is used for one-shot learning. It means you don't need a huge dataset for it's training, one example in each class is enough for it's working, though in this implementation more thatn one example has been used but it is still very less compared to what the state of art algorithm requires.

## Getting Started

Git clone the repository and run the ipynb file in Google Colab or Jupyter Notebook(some importing changes will be required).

### Prerequisites

Google Colab: No installations required

Jupyter Notebook: Install Pytorch and torchvision using pip

```
pip install pytorch
pip install torchvision
```

## Dataset

AT & T. You can get the dataset by cloning this repository

## Running the tests

You can run the code to train the model and save it for later use. Just uncomment the loading model part to use the pre-trained model.
Or you can download the pretrained model from [Here](https://drive.google.com/open?id=15YCXIv1Y2uSQJAENxRFjtXq1fQgEX53v) though keep in mind that it is a GPU model so it won't work on a CPU. If you're using this pretrained model no need to run the training part of the code. You can directly load the model and test the code.


## Built With

* [Pytorch](https://pytorch.org/) - The Deep Learning library used


## Authors

* **Parth Goel** - *Initial work* - [parthgoe1](https://github.com/parthgoe1)
* **Vaibhav Singh** - *Initial work* - [singhv04](https://github.com/singhv04)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.


## References

* https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch
* https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

