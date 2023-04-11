import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    test loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_test_loss=float('inf')):
        self.best_test_loss = best_test_loss
        
    def __call__(self, current_test_loss, epoch, model, optimizer, criterion):

        if current_test_loss < self.best_test_loss:
            self.best_test_loss = current_test_loss
            print(f"Best test loss: {self.best_test_loss}")
            print(f"Saving the model ........")
            
            torch.save(model,"artifacts/model.pkl")




def save_plots(train_acc, test_acc, train_loss, test_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        test_acc, color='blue', linestyle='-', 
        label='test accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('artifacts/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        test_loss, color='red', linestyle='-', 
        label='test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('artifacts/loss.png')