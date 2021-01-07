# FaceRecognitionModel
A face recognition model with resNet-18 backbone

HOW TO RUN:
The model was developed in FaceRec.ipynb.ipynb on AWS's ec2 server. The file can be run by executing the cells sequentially on a jupyter notebook or in colab.

Model description:
A classification based model was trained comprising of the following elements:

1. Data Loading
- torchvision.datasets.ImageFolder was used to create the Image datasets.
- This was done for classification_data/train_data, classification_data/val_data, classification_data/test_data, and verification data.
- Simple dataLoader class was used to create data loaders from the aforementioned datasets, with a batch size of 64, and num_workers = 4.
- In __init__, the indices of the individual data samples in each utterance are stored in dictionaries to allow O(1) access in __getitem__().
  So, each data sample index (key) maps to a tuple of the utterance index 'i' and row within ith utterance index 'j'. (ind -> (i,j))
- Each utterance is then padded to make the input consistent with the padding scheme. The labels are not padded
- __len__ return the number of data samples (total number of rows in all utterances combined) without padding
- __getitem__ accesses the utterance index 'i' and element index 'j' mapped to 'index' in the dictionary created in __init__.
  It takes 'context' number of rows before and after the selected row and passes this sliced array as a tensor. A single label is returned corresponding to each tensor input.


2. Network Architecture
- I used ResNet-18 Architecture defined in the ResNet paper as the CNN. AlexNet and ResNet-34 were also experimented with. ResNet-18 gave the best performance to time ratio. 
- The Basic block implementation is a slight modification of the code shared by Bharat.
- The first block of the network comprises of 4 layers, namely a Conv2d, a batchnorm2d, a ReLU and a MaxPool2d. Parameters of the layers were taken from PyTorch's implementation of ResNet-18. 
- This is followed by 8 basic blocks each comprising of a Conv2d, a BatchNorm2d, a ReLU, another Conv2d, and a BatchNorm2d.
- Shortcuts are introduced at the end of every Basic block. Every shortcut block comprised of Conv2d and BatchNorm2d layer.
- Following the ResNet paper, the network has a AvgPool2d layer at the end followed by a linear layer with output features equal to the number of classes (4000).
- The classification is handled by the CrossEntropyLoss criterion which contains a Softmax unit followed by a cross-entropy loss unit for multi-class classification.
- The network has 2 forward methods - one for training the classifier and the other for generating the embeddings. The latter does not use the linear layer mentioned above.


3. Hyperparameters and implementation details
- epochs = 15
  I ran the model for 15 epochs and checked the validation accuracy every 2 epochs. 
  The validation accuracy was increasing even at the end of 15 epochs thus indicating no overfitting.

- training batch_size = 64
  Batch sizes of 32 and 64, and 124 were experimented with. Increasing the batch size sped up the computation due to vectorized implementation.
  With a batch size of 124 and above however, the GPU ran out of memory. The model was thus trained with a batch size of 64.

- Optimizer = SGD
  SGD with default learning rate of 0.15 was used to optimize the weight updates.
  The learning rate was annealed manually by a factor of 0.85 every epoch. This seemed to work pretty well, so I didn't try using an LR scheduler.

- Criterion = CrossEntropyLoss
  The cross entropy loss criterion was used since it is the go-to criterion for multi-class classification problems.
  No other criteria were experimented with for a lack of time.

- Similarity Measure
  The cosine similarity measure provided by PyTorch was used to compute the similarity score between embeddings.
  
- Preprocessing
  The images were rescaled from to 256x256 and then cropped at the center to 224x224 to fit ResNet-18's architecture.
  The images were also normalized using the mean and std from the ImageNet dataset.

