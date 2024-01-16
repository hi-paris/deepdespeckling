from glob import glob
import os
import torch
from deepdespeckling.merlin.training.GenerateDataset import GenerateDataset

from deepdespeckling.merlin.training.Dataset import Dataset, ValDataset
from deepdespeckling.merlin.training.model import Model
from deepdespeckling.utils.constants import PATCH_SIZE


def evaluate(model, loader):
    outputs = [model.validation_step(batch) for batch in loader]
    outputs = torch.tensor(outputs).T
    loss, accuracy = torch.mean(outputs, dim=1)
    return {"loss": loss.item(), "accuracy": accuracy.item()}


def save_model(model, destination_folder):
    """Save the given model to the given destination folder

    Args:
        model (torch model): a trained torch model
        destination_folder (str): path of a folder where to store the model
    """
    # Check whether the specified path exists or not
    isExist = os.path.exists(destination_folder)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(destination_folder)
        print("The new directory is created!")
        torch.save(model.state_dict(), destination_folder+"/model.pth")
    else:
        torch.save(model.state_dict(), destination_folder+"/model.pth")


def fit(model, train_loader, val_loader, epochs, lr_list, eval_files, eval_set, clip_by_norm, sample_dir):
    """ Fit the model according to the given evaluation data and parameters.

    Parameters
    ----------
      model : model as defined in main
      train_loader : Pytorch's DataLoader of training data
      val_loader : Pytorch's DataLoader of validation data
      lr_list : list of learning rates
      eval_files : .npy files used for evaluation in training
      eval_set : directory of dataset used for evaluation in training

    Returns
    ----------
      self : object
        Fitted estimator.

    """

    train_losses = []
    history = {}
    epoch_num = 0
    for epoch in range(epochs):
        epoch_num = epoch_num + 1
        print("\nEpoch", epoch + 1)
        print("***************** \n")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[epoch])

        # Train
        for i, batch in enumerate(train_loader, 0):
            optimizer.zero_grad()
            loss = model.training_step(batch, i)
            train_losses.append(loss)

            loss.backward()

            if (clip_by_norm == True):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0)

            optimizer.step()

            # running_loss += loss.item()     # extract the loss value
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))
            # zero the loss

        # Validate
        with torch.no_grad():
            image_num = 0
            for batch in val_loader:
                val_loss = model.validation_step(
                    batch, image_num, epoch_num, eval_files, eval_set, sample_dir)
                image_num = image_num+1

    history["train_loss"] = train_losses

    return history


def create_model(batch_size=12, val_batch_size=1, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 from_pretrained=False):
    """ Runs the denoiser algorithm for the training and evaluation dataset


    """
    this_dir, this_filename = os.path.split(__file__)
    weights_path = os.path.join(this_dir, "saved_model", "model.pth")

    model = Model(batch_size, val_batch_size, device)
    model.to(device)

    if from_pretrained == True:
        model.load_state_dict(torch.load(
            weights_path, map_location=torch.device('cpu')))

    return model


def fit_model(model, lr_list, nb_epoch, training_set_directory, validation_set_directory, sample_directory,
              save_directory, patch_size=PATCH_SIZE, batch_size=12, val_batch_size=1, stride_size=128,
              n_data_augmentation=1, seed=2, clip_by_norm=True):
    """ Runs the denoiser algorithm for the training and evaluation dataset

    Parameters
    ----------
    model : model as defined in main
    lr_list : list of learning rates

    Returns
    ----------
    history : list of both training and validation loss

    """
    torch.manual_seed(seed)

    train_data = GenerateDataset().generate_patches(src_dir=training_set_directory, patch_size=patch_size, step=0,
                                                    stride=stride_size, bat_size=batch_size, data_aug_times=n_data_augmentation)

    # Prepare train DataLoader
    train_dataset = Dataset(train_data)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Prepare Validation DataLoader
    eval_dataset = ValDataset(validation_set_directory)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=val_batch_size, shuffle=False, drop_last=True)

    eval_files = glob(validation_set_directory+'/*.npy')
    # Train the model
    history = fit(model, train_loader, eval_loader, nb_epoch, lr_list, eval_files, validation_set_directory,
                  clip_by_norm, sample_directory)

    # Save the model
    save_model(model, save_directory)
    print("\n model saved at :", save_directory)

    return history
