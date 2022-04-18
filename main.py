import torch
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, sampler

DOG_TRAINING_SET = "./dogvscat/train/1/*.jpg"
CAT_TRAINING_SET = "./dogvscat/train/0/*.jpg"
DOG_TEST_SET = "./dogvscat/test/1/*.jpg"
CAT_TEST_SET = "./dogvscat/test/1/*.jpg"
class Binary_Classifier():
    def __init__(self):
        # Load training, validation and test data
        self.X_train = []
        self.Y_train = []
        self.X_validation = []
        self.Y_validation = []
        self.X_test = []
        self.Y_test = []

        self.get_data()

        # Scale the features to a standard normal
        self.X_train = self.scaler(self.X_train)
        self.X_validation = self.scaler(self.X_validation)
        self.X_test = self.scaler(self.X_test)

        # Datasets from folders
        batch_size = 64
        # Transfer the data from numpy to tensor
        self.data = {
            'train': TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.Y_train).to(torch.long)),
            'valid': TensorDataset(torch.from_numpy(self.X_validation).float(), torch.from_numpy(self.Y_validation).to(torch.long))
        }

        # Dataloader iterators
        self.dataloaders = {
            'train': DataLoader(self.data['train'], batch_size = batch_size, shuffle = True, num_workers = 2),
            'valid': DataLoader(self.data['valid'], batch_size = batch_size, shuffle = False, num_workers = 2)
        }

        trainiter = iter(self.dataloaders['train'])
        features, labels = next(trainiter)
        features.shape, labels.shape


        #hyper parameters
        self.learning_rate = 0.001
        self.epochs = 500
        # Model , Optimizer, Loss
        input_shape = [np.array(self.X_train).shape[1], np.array(self.X_train).shape[2], np.array(self.X_train).shape[3]]
        self.model = model.CNN_Model(input_shape)
        self.optimizer = torch.optim.SGD(self.model.cuda().parameters(), lr = learning_rate, momentum = 0.9)
        self.criterion = nn.CrossEntropyLoss()

        for p in self.optimizer.param_groups[0]['params']:
            if p.requires_grad:
                print(p.shape)



    def get_data(self):
        for dog_training in glob.glob(DOG_TRAINING_SET):
            image = cv2.imread(dog_training)
            if ((image.shape[1], image.shape[0]) > (256, 256)):
                image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
            elif ((image.shape[1], image.shape[0]) < (256, 256)):
                image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)
            # The total number of training images is 500, so 400 is taken for training set and 100 for validation
            # This is the dog set so we take 400 / 2 = 200     
            if (np.array(self.X_train).shape[0] < 200):
                self.X_train.append(image)
                self.Y_train.append(1)
            else:
                self.X_validation.append(image)
                self.Y_validation.append(1)
        for cat_training in glob.glob(CAT_TRAINING_SET):
            image = cv2.imread(cat_training)
            if ((image.shape[1], image.shape[0]) > (256, 256)):
                image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
            elif ((image.shape[1], image.shape[0]) < (256, 256)):
                image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)   
            # This is the cat set so we take the last 400 / 2 = 200
            if (np.array(self.X_train).shape[0] < 400):
                self.X_train.append(image)
                self.Y_train.append(0)
            else:
                self.X_validation.append(image)
                self.Y_validation.append(0)
        for dog_test in glob.glob(DOG_TEST_SET):
            image = cv2.imread(dog_test)  
            if ((image.shape[1], image.shape[0]) > (256, 256)):
                image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
            elif ((image.shape[1], image.shape[0]) < (256, 256)):
                image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)
            self.X_test.append(image)
            self.Y_test.append(1)
        for cat_test in glob.glob(CAT_TEST_SET):
            image = cv2.imread(cat_test)  
            if ((image.shape[1], image.shape[0]) > (256, 256)):
                image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
            elif ((image.shape[1], image.shape[0]) < (256, 256)):
                image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)  
            self.X_test.append(image)
            self.Y_test.append(0)

        self.X_train = np.array(self.X_train)
        self.X_validation = np.array(self.X_validation)
        self.X_test = np.array(self.X_test)
        self.Y_train = np.array(self.Y_train)
        self.Y_validation = np.array(self.Y_validation)
        self.Y_test = np.array(self.Y_test)

        print('X_train shape:', self.X_train.shape)
        print('X_test shape:', self.X_test.shape)
        print('X_validation shape:', self.X_validation.shape)
        print('Y_validation shape:', self.Y_validation.shape)
        print('Y_train shape:', self.Y_train.shape)
        print('Y_test shape:', self.Y_test.shape)
    def scaler(self, array):
        array_transformed = np.zeros_like(array)
        for i in range(np.array(array).shape[3]):
            sc = StandardScaler()
            slc = array[:, :, :, i].reshape(np.array(array).shape[0], np.array(array).shape[1] * np.array(array).shape[2]) # make it a bunch of row vectors
            transformed = sc.fit_transform(slc)
            transformed = transformed.reshape(np.array(array).shape[0], np.array(array).shape[1],  np.array(array).shape[2]) # reshape it back to tiles
            array_transformed[:, :, :, i] = transformed # put it in the transformed array
        return array_transformed

    #def process_image(self, filepath):

    # def convert_and_pubish(self):

    def dog_cat_classifier(self):
        save_file_name = f'dog_v_cat_cnn_model.pt'
        train_on_gpu = cuda.is_available()
        history = self.train(save_file_name = save_file_name, max_epochs_stop = 3, n_epochs = 500, print_every = 1)

        plt.figure(figsize=(8, 6))
        for c in ['train_loss', 'valid_loss']:
            plt.plot(
                history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Negative Log Likelihood')
        plt.title('Training and Validation Losses')
        plt.show()

        plt.figure(figsize=(8, 6))
        for c in ['train_acc', 'valid_acc']:
            plt.plot(
                100 * history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.show()

    def train(save_file_name,
        max_epochs_stop = 3,
        n_epochs = 500,
        print_every = 1):
        """Train a PyTorch Model

        Params
        --------
            save_file_name (str ending in '.pt'): file path to save the model state dict
            max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
            n_epochs (int): maximum number of training epochs
            print_every (int): frequency of epochs to print training stats

        Returns
        --------
            history (DataFrame): history of train and validation loss and accuracy
        """

        # Early stopping intialization
        epochs_no_improve = 0
        valid_loss_min = np.Inf

        valid_max_acc = 0
        history = []

        # Number of epochs already trained (if using loaded in model weights)
        try:
            print(f'Model has been trained for: {model.epochs} epochs.\n')
        except:
            self.model.epochs = 0
            print(f'Starting Training from Scratch.\n')

        overall_start = timer()
        device = torch.device('cuda')
        # Main loop
        for epoch in range(n_epochs):

            # keep track of training and validation loss each epoch
            train_loss = 0.0
            valid_loss = 0.0

            train_acc = 0
            valid_acc = 0

            # Set to training
            self.model.train()

            start = timer()

            # Training loop
            for ii, (data, target) in enumerate(self.dataloaders['train']):
                
                # Tensors to gpu, both model parameters, data, and target need to be tensors.
                # You can use .cuda() function
                data = data.to(device)
                target = target.to(device)
                # Clear gradients
                self.optimizer.zero_grad()

                # Forward path
                output = self.model(data)

                # Loss function 
                loss = self.criterion(output, target)            

                # Backward path (backpropagation)
                loss.backward()            

                # Update the parameters
                self.optimizer.step()

                # Track train loss by multiplying average loss by number of examples in batch
                train_loss += loss.item() * data.size(0)

                # Calculate accuracy by finding max log probability
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))

                # Need to convert correct tensor from int to float to average
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

                # Multiply average accuracy times the number of examples in batch
                train_acc += accuracy.item() * data.size(0)

                # Track training progress
                print(
                    f'Epoch: {epoch}\t{100 * (ii + 1) / len(self.train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')

            # After training loops ends, start validation
            self.model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():

                # Set to evaluation mode
                self.model.eval()

                # Validation loop
                for (data, target) in self.dataloaders['valid']:
                    # Tensors to gpu
                    data = data.to(device)
                    target = target.to(device)

                    # Forward path
                    output = self.model(data)

                    # Validation loss computation
                    loss = self.criterion(output, target)

                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                    correct_tensor.type(torch.FloatTensor))

                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)


            # Calculate average losses and Calculate average accuracy
            train_loss = train_loss / len(self.train_loader.dataset)
            valid_loss = valid_loss / len(self.valid_loader.dataset)

            train_acc = train_acc / len(self.train_loader.dataset)
            valid_acc = valid_acc / len(self.valid_loader.dataset)

            history.append([train_loss, valid_loss, train_acc, valid_acc])

            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                print(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                )
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                )

            print(valid_loss)
            # Save the model if validation loss decreases
            if valid_loss < valid_loss_min:
                # Save model 
                # You can use torch.save()
                torch.save(self.model.state_dict(), save_file_name)

                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valid_loss
                valid_best_acc = valid_acc
                best_epoch = epoch

            # Otherwise increment count of epochs with no improvement
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                    )
                    total_time = timer() - overall_start
                    print(
                        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                    )

                    # Load the best state dict
                    # You can use model.load_state_dict()
                    self.model.load_state_dict(torch.load(save_file_name))

                    # Attach the optimizer
                    self.model.optimizer = self.optimizer

                    # Format history
                    history = pd.DataFrame(
                        history,
                        columns=[
                            'train_loss', 'valid_loss', 'train_acc',
                            'valid_acc'
                        ])
                    return history

        # Attach the optimizer
        self.model.optimizer = self.optimizer
        # Record overall time and print out stats
        total_time = timer() - overall_start
        print(
            f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%'
        )
        print(
            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
        )
        # Format history
        history = pd.DataFrame(
            history,
            columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
        return history

    # def YUVtoRGB(self):

if __name__ == '__main__':
    launch = Binary_Classifier()
    launch.dog_cat_classifier()