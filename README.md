## IMU-TL: Transfer Learning for Inertial-based Activity Recognition 
This repository provides a framework for evaluating the benefits and limitations of Transfer Learning for 
Inertial-based Activity Recognition. 

### Input
The entry point is ```main.py```. This script execpts an input (dataset file) in the form of tab or comma delimited file, where the first N columns give the IMU signal (N=6 for acc and gyro) and the N+1 column gives the class id

### Learning with imu-tl
**Training**

```python main.py train imu-cnn <path to dataset file> <path to config file>```

**Transfer Learning (initialization from weights)**
```
python main.py transfer imu-cnn <path to dataset file> <path to config file> --pretrained_path <path to trained model>
```

**Transfer Learning (fine-tuning classifier head only)**
```
python main.py transfer imu-cnn <path to dataset file> <path to config file> --pretrained_path <path to trained model> --finetune
```

**Testing**
``` 
python main.py test imu-cnn <path to dataset file> <path to config file> --pretrained_path <path to trained model>
```

For training and testing a *transformer-based architecture* change from 'imu-cnn' to 'imu-transformer'
The training and architecture hyper-parameters are controlled by the config file. See more details below 

### Configuration Parameters
Parameter Name | Description |
--- | --- |
General parameters|
n_freq_print|How often to print the loss to the log file
n_freq_checkpoint|How often to save a checkpoint
n_workers|Number of workers
device_id|The identifier of the torch device (e.g., cuda:0)
Data parameters|
input_dim|The dimension of the input IMU data, e.g., 6 when using accelerometers and gyros
window_size|The size of the time window (i.e. how many samples in a window)
num_classes|Number of classes
window_shift|The window shift, put here the window_size to avoid window overlap
Training hyper-parameters|
batch_size| The batch size
lr|The learning rate
weight_decay|The weight decay 
eps| epsilon for Adam optimizer
lr_scheduler_step_size|How often to decay the learning rate
lr_scheduler_gamma|By what factor to decay the learning rate
n_epochs|Number of epochs
Transformer architecture hyper-parameters|
encode_position|Whether to encode positions for IMU-Transformer
transformer_dim|The latent dimension of the Transformer Encoder
nhead|Number of heads for the MHA operation
num_encoder_layers| Number of encoder layers (Transformer)
dim_feedforward:| The latent dimension of the Encoder's MLP
transformer_dropout| The dropout applied to the Transformer
transformer_activation| Activation function used for the Transformer (gelu/relu/elu)
head_activation|Activation function used for the MLP classifier head 
CNN architecture hyper-parameters|
dropout| Dropout applied for the CNN model
latent_dim| Dimension of the latent convolution layer 
