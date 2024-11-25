import os
import copy
import torch
import random
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns


from datetime import datetime
from importlib import import_module
from torch.utils.data import DataLoader
import torch.nn.utils as nn_utils
from sklearn.metrics import classification_report, confusion_matrix

from .modules.utils import filter_kwargs_for_module


class Classifier:
    def __init__(self,
                 model_name: str,
                 model: nn.Module,
                 num_classes: int,
                 output_base: str,
                 output_name: str ,
                 class_weight,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 criterion=None,
                 optimizer=optim.Adam,
                 scheduler=None,
                 classes=None,
                 ):

        self.num_classes = num_classes
        if classes is None:
            self.classes = [str(i) for i in range(num_classes)]
        else:
            self.classes = classes

        if criterion == None:

            if class_weight != None:

                class_weight = class_weight.to(device)
                criterion = nn.CrossEntropyLoss(weight =class_weight)
            else:

                criterion == nn.CrossEntropyLoss()
        if num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()

        self.model_name = model_name
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = scheduler
        self.outout_base = os.path.join(output_base ,"model_output")
        self.best_model_state_dict = None

        nowtime = datetime.now().strftime("%Y_%m_%d %H:%M")
        if output_name == None:
            self.output_dir = os.path.join(self.outout_base, self.model_name, nowtime)
        else:
            self.output_dir = os.path.join(self.outout_base, self.model_name, output_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.training_loss_list = []
        self.validation_loss_list = []
        self.training_accuracy_list = []
        self.validation_accuracy_list = []

    def train(self,
              epochs: int,
              train_loader: DataLoader,
              val_loader=None,
              test_loader=None,
              save_model=True,
              save_checkpoint=True,
              save_strategy="val_accuracy",
              save_best_only=False,
              lr=None):
        try:
            # print(f"************* {self.criterion}*****************")
            self.train_model(epochs, train_loader, val_loader, test_loader, save_model, save_checkpoint, save_strategy,
                             save_best_only, lr)
        except KeyboardInterrupt:
            print("Training interrupted. Saving the checkpoint...")
            self.save_checkpoint(os.path.join(self.output_dir, "interrupted.ckpt"))
            print("Checkpoint saved.")

    def train_model(self,
                    epochs: int,
                    train_loader: DataLoader,
                    val_loader=None,
                    test_loader=None,
                    save_model=True,
                    save_checkpoint=True,
                    save_strategy="val_accuracy",
                    save_best_only=False,
                    lr=None) -> None:

        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        if save_model:
            model_dir = os.path.join(self.output_dir, "weights")
            os.makedirs(model_dir, exist_ok=True)

        if save_checkpoint:
            checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

        if save_strategy in ["val_loss", "val_accuracy"] and val_loader is None:
            raise ValueError("Cannot use 'val_loss' or 'val_accuracy' as save_strategy without validation data.")
        if save_strategy not in ["train_loss", "val_loss", "train_accuracy", "val_accuracy", "every_epoch"]:
            raise ValueError(
                "Invalid 'save_strategy'. Must be one of ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'every_epoch'].")

        best_training_loss = float("inf")
        best_validation_loss = float("inf")
        best_training_accuracy = 0.0
        best_validation_accuracy = 0.0

        plot_dir = os.path.join(self.output_dir, "graphs")
        os.makedirs(plot_dir, exist_ok=True)

        # print("-- Start Training --")
        # for epoch in range(epochs):
        for epoch in tqdm(range(epochs), desc="Training Epochs"):

            if save_model and save_best_only:
                model_save_path = os.path.join(model_dir, f"model.pth")
            else:
                model_save_path = os.path.join(model_dir, f"epoch_{epoch + 1}.pth")

            self.model.train()
            training_loss = 0.0
            correct = 0.0
            total_samples = 0.0
            # for inputs, labels in train_loader:
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                # inputs = torch.tensor(inputs, dtype=torch.float32) Warning
                inputs = inputs.float()
                labels = labels.long()
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                # **************************************************************#
                if torch.isnan(outputs).any():
                    print("NaN detected in outputs")
                # **************************************************************#
                loss = self.criterion(outputs, labels)
                # **************************************************************#
                if torch.isnan(loss).any():
                    print("NaN detected in loss")
                # **************************************************************#
                loss.backward()
                # **************************************************************#
                # Apply gradient clipping
                clip_value = 1.0
                nn_utils.clip_grad_norm_(self.model.parameters(), clip_value)
                # **************************************************************#

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                training_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            epoch_training_loss = training_loss / len(train_loader)
            epoch_training_accuracy = correct / total_samples

            self.training_loss_list.append(epoch_training_loss)
            self.training_accuracy_list.append(epoch_training_accuracy)

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_correct = 0.0
                    val_total_samples = 0.0
                    # for inputs, labels in val_loader:
                    for inputs, labels in tqdm(val_loader, desc="Validation ... ", leave=False):
                        inputs = inputs.float()

                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total_samples += labels.size(0)

                    epoch_val_loss = val_loss / len(val_loader)
                    epoch_val_accuracy = val_correct / val_total_samples

                    self.validation_loss_list.append(epoch_val_loss)
                    self.validation_accuracy_list.append(epoch_val_accuracy)

                    print(
                        f"[Epoch {epoch + 1}/{epochs}] training_loss: {epoch_training_loss:.12f} training_accuracy: {epoch_training_accuracy:.12f} val_loss: {epoch_val_loss:.12f} val_accuracy: {epoch_val_accuracy:.12f}")

                if save_model:
                    if save_strategy == "train_loss":
                        if epoch_training_loss < best_training_loss:
                            # torch.save(self.model.state_dict(), model_save_path)
                            self.save_model(model_save_path)
                            best_training_loss = epoch_training_loss
                    elif save_strategy == "val_loss":
                        if epoch_val_loss < best_validation_loss:
                            # torch.save(self.model.state_dict(), model_save_path)
                            self.save_model(model_save_path)
                            best_validation_loss = epoch_val_loss
                    elif save_strategy == "train_accuracy":
                        if epoch_training_accuracy > best_training_accuracy:
                            # torch.save(self.model.state_dict(), model_save_path)
                            self.save_model(model_save_path)
                            best_training_accuracy = epoch_training_accuracy
                    elif save_strategy == "val_accuracy":
                        if epoch_val_accuracy > best_validation_accuracy:
                            # torch.save(self.model.state_dict(), model_save_path)
                            self.save_model(model_save_path)
                            best_validation_accuracy = epoch_val_accuracy
                    elif save_strategy == "every_epoch":
                        # torch.save(self.model.state_dict(), model_save_path)
                        self.save_model(model_save_path)


                self.plot_loss(plot_dir, mode="both")
                self.plot_accuracy(plot_dir, mode="both")

            else:
                print(
                    f"[Epoch {epoch + 1}/{epochs}] training_loss: {epoch_training_loss:.12f} training_accuracy: {epoch_training_accuracy:.12f}")
                if save_model:
                    if save_strategy == "train_loss":
                        if epoch_training_loss < best_training_loss:
                            # torch.save(self.model.state_dict(), model_save_path)
                            self.save_model(model_save_path)
                            best_training_loss = epoch_training_loss
                    elif save_strategy == "train_accuracy":
                        if epoch_training_accuracy > best_training_accuracy:
                            # torch.save(self.model.state_dict(), model_save_path)
                            self.save_model(model_save_path)
                            best_training_accuracy = epoch_training_accuracy
                    elif save_strategy == "every_epoch":
                        # torch.save(self.model.state_dict(), model_save_path)
                        self.save_model(model_save_path)

                self.plot_loss(plot_dir, mode="training")
                self.plot_accuracy(plot_dir, mode="training")

            if save_checkpoint:
                checkpoint_save_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.ckpt")
                self.save_checkpoint(checkpoint_save_path)

        self.save_training_history(self.output_dir)
        print("-- Finished Training --")

        if test_loader is not None:
            self.evaluate(test_loader, return_report=True, return_confusion_matrix=True)

    def predict(self, data: torch.Tensor, as_numpy=False):

        if len(data.shape) == 2:
            data = data.reshape(-1, data.shape[0], data.shape[1]).float()
        elif len(data.shape) == 3:
            data = data.float()
        else:
            raise ValueError("Invalid input shape. Must be 2D or 3D tensor.")

        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)

        if as_numpy:
            return predicted.cpu().numpy(), outputs.cpu().numpy()

        return predicted, outputs

    def evaluate(self,
                 data_loader: DataLoader,
                 return_report=False,
                 return_confusion_matrix=False,
                 with_best_model=False):

        # print("-- Evaluation --")

        if with_best_model and self.best_model_state_dict is not None:
            # print(self.model.state_dict())
            self.model.load_state_dict(self.best_model_state_dict)
            # print(self.model.state_dict())
            print("Loaded the best model.")

        self.model.eval()
        correct = 0.0
        total = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            # for inputs, labels in data_loader:
            for inputs, labels in tqdm(data_loader, desc="-- Evaluation --", leave=False):
                inputs = inputs.float()

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        accuracy = correct / total

        print(f"Accuracy: {accuracy}")

        if return_report:
            # print(f'******* \n self.classes {self.classes} \n *******')
            report = classification_report(y_true, y_pred, target_names=self.classes , labels= list(range(len(self.classes))))
            print("Classification Scores:\n", report)

        if return_confusion_matrix:
            cm, cm_reg = self.make_confusion_matrix(y_true, y_pred , self.classes)
            # print("Confusion Matrix:\n", cm)
            print("Regularized Confusion Matrix:\n", cm_reg)

    def make_confusion_matrix(self,
                              y_true,
                              y_pred,
                              class_names , figsize=(18, 12),
                              ):

        cm = confusion_matrix(y_true, y_pred)
        cm_reg = np.array([[j / sum(i) for j in i] for i in cm])

        plt.figure(figsize=figsize)
        plt.figure()
        # sns.heatmap(cm_reg,
        #             cbar=True,
        #             annot=True,
        #             fmt=".2f",
        #             cmap=cmap,
        #             xticklabels=self.classes,
        #             yticklabels=self.classes,
        #             annot_kws={'fontsize': 20},
        #             vmax=1.0,
        #             vmin=0.0)
        sns.heatmap(cm_reg, annot=True, fmt='.2f',cmap='Blues', xticklabels=class_names, yticklabels=class_names, vmax=1.0,
                    vmin=0.0)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"), format="png")
        plt.close()

        return cm, cm_reg
    # def make_confusion_matrix(self, y_true, y_pred, class_names ,figsize=(10, 10)):
    #     cm = confusion_matrix(y_true, y_pred,labels=np.arange(len(class_names)))
    #     cm_reg = np.array([[j / sum(i) if sum(i) > 0 else 0 for j in i] for i in cm])
    #
    #     # Create a dot plot style confusion matrix
    #     plt.figure(figsize=figsize)
    #     plt.title('Confusion Matrix (Dot Plot)', fontsize=16)
    #
    #     # Get the number of classes
    #     n_classes = len(class_names)
    #
    #     # Create the grid
    #     x = np.arange(n_classes)
    #     y = np.arange(n_classes)
    #
    #     # Create the grid with appropriate spacing
    #     plt.xticks(x, class_names, rotation=45, ha="right")
    #     plt.yticks(y, class_names)
    #
    #     # Plot the dots for each cell in the confusion matrix
    #     for i in range(n_classes):
    #         for j in range(n_classes):
    #             plt.scatter(j, i, s=cm_reg[i, j] * 3000, color='blue', alpha=0.6)  # The size is proportional to cm_reg
    #
    #             # Annotate the percentage in the middle of each circle
    #             plt.text(j, i, f'{cm_reg[i, j]:.2f}', ha='center', va='center', fontsize=12, color='white')
    #
    #     plt.xlabel("Predicted Label")
    #     plt.ylabel("True Label")
    #
    #     # Invert the y-axis to match heatmap style (origin at top-left)
    #     plt.gca().invert_yaxis()
    #     plt.gca().set_aspect('equal', adjustable='box')
    #
    #     # Save the plot
    #     plt.savefig(os.path.join(self.output_dir, "confusion_matrix_dot_plot.png"), format="png")
    #     plt.close()
    #
    #     return cm, cm_reg

    def save_model(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), save_path)
        self.best_model_state_dict = copy.deepcopy(self.model.state_dict())

    def load_model(self, model_path: str, model_key="model") -> None:
        model = torch.load(model_path)
        if model.__class__ == dict:
            self.model.load_state_dict(model[model_key])
        else:
            self.model.load_state_dict(model)
        self.model.to(self.device)

    def save_checkpoint(self, save_path: str) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            "torch_cuda_random": torch.cuda.get_rng_state(),
            "torch_cuda_random_all": torch.cuda.get_rng_state_all(),
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str, history_path=None) -> None:
        checkpoint = torch.load(checkpoint_path)
        if checkpoint.__class__ == dict:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint["scheduler"] is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["numpy"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["torch_cuda_random"])
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_all"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        if history_path is not None:
            self.load_training_history(history_path)

    def plot_loss(self,
                  save_dir: str,
                  mode="both",
                  figsize=(10, 10),
                  colors=('b-', 'r-')):

        if mode not in ["both", "training", "validation"]:
            raise ValueError("Invalid mode. Must be one of ['both', 'training', 'validation']")

        if self.validation_loss_list == []:
            mode = "training"

        epochs = range(1, len(self.training_loss_list) + 1)

        # plt.figure(figsize=figsize)
        plt.figure()
        if mode == "both":
            training_loss = self.training_loss_list
            validation_loss = self.validation_loss_list
            plt.plot(epochs, training_loss, colors[0], label='Training Loss')
            plt.plot(epochs, validation_loss, colors[1], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(save_dir, "training_validation_loss.png"))
        elif mode == "training":
            training_loss = self.training_loss_list
            plt.plot(epochs, training_loss, colors[0], label='Training Loss')
            plt.title('Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(save_dir, "training_loss.png"))
        elif mode == "validation":
            validation_loss = self.validation_loss_list
            plt.plot(epochs, validation_loss, colors[1], label='Validation Loss')
            plt.title('Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(save_dir, "validation_loss.png"))
        plt.close()

    def plot_accuracy(self,
                      save_dir: str,
                      mode="both",
                      figsize=(18, 12),
                      colors=('b-', 'r-')):

        if mode not in ["both", "training", "validation"]:
            raise ValueError("Invalid mode. Must be one of ['both', 'training', 'validation']")

        if self.validation_accuracy_list == []:
            mode = "training"

        epochs = range(1, len(self.training_accuracy_list) + 1)

        plt.figure()
        if mode == "both":
            training_accuracy = self.training_accuracy_list
            validation_accuracy = self.validation_accuracy_list
            plt.plot(epochs, training_accuracy, colors[0], label='Training Accuracy')
            plt.plot(epochs, validation_accuracy, colors[1], label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(os.path.join(save_dir, "training_validation_accuracy.png"))
        elif mode == "training":
            training_accuracy = self.training_accuracy_list
            plt.plot(epochs, training_accuracy, colors[0], label='Training Accuracy')
            plt.title('Training Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(os.path.join(save_dir, "training_accuracy.png"))
        elif mode == "validation":
            validation_accuracy = self.validation_accuracy_list
            plt.plot(epochs, validation_accuracy, colors[1], label='Validation Accuracy')
            plt.title('Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(os.path.join(save_dir, "validation_accuracy.png"))
        plt.close()

    def save_training_history(self, save_dir: str) -> None:
        # save as .npz file
        np.savez_compressed(os.path.join(save_dir, "training_history.npz"),
                            training_loss=self.training_loss_list,
                            validation_loss=self.validation_loss_list,
                            training_accuracy=self.training_accuracy_list,
                            validation_accuracy=self.validation_accuracy_list)

    def load_training_history(self, history_path: str) -> None:
        history = np.load(history_path)
        # set the parameters as list
        self.training_loss_list = list(history["training_loss"])
        self.validation_loss_list = list(history["validation_loss"])
        self.training_accuracy_list = list(history["training_accuracy"])
        self.validation_accuracy_list = list(history["validation_accuracy"])


def build_classifier(model_name: str,
                     timestep: int,
                     dimentions: int,
                     num_classes: int,
                     output_base:str,
                     output_name = None,
                     model_path=None,
                     history_path=None,
                     num_features=256,
                     dropout_rate=0.3,
                     num_layers=2,
                     custom_head=None,
                     custom_modules_dir=None,
                     class_labels=None,
                     class_weight = None,
                     **kwargs) -> Classifier:
    package_dir, registry = make_model_registry(custom_modules_dir)
    if model_name not in registry:
        raise ValueError(f"Invalid model name. Must be one of {list(registry.keys())}")
    builder_class = import_module_from_string(registry[model_name], package=package_dir)
    builder_kwargs = filter_kwargs_for_module(builder_class, **kwargs)

    # print(builder_kwargs)

    builder = builder_class(timestep=timestep,
                            dimentions=dimentions,
                            num_features=num_features,
                            num_classes=num_classes,
                            dropout_rate=dropout_rate,
                            num_layers=num_layers,
                            custom_head=custom_head,
                            **builder_kwargs)
    model = builder.build()

    classifier = Classifier(model_name,
                            model,
                            num_classes,
                            output_base,
                            output_name,
                            class_weight,
                            classes=class_labels)

    if model_path is not None:
        classifier.load_checkpoint(model_path)
    if history_path is not None:
        classifier.load_training_history(history_path)

    return classifier


# function to make a model registry
def make_model_registry(custom_modules_dir=None):
    if custom_modules_dir is None:
        module_dir = os.path.join(os.path.dirname(__file__), "modules")
        dot = "."
    else:
        module_dir = custom_modules_dir
        dot = ""
    model_registry = {}
    for module in os.listdir(module_dir):
        if module == '__init__.py' or module[-3:] != '.py':
            continue
        model_name = module[:-3]
        model_registry[
            model_name] = f"{dot}{os.path.basename(module_dir).replace('/', '.')}.{model_name}.{model_name}Builder"
    package_dir = os.path.basename(os.path.dirname(__file__))
    return package_dir, model_registry


# function to dynamically import a module
def import_module_from_string(module_string: str, package="tisc"):
    module_path, module_name = module_string.rsplit('.', 1)
    module = import_module(module_path, package=package)
    return getattr(module, module_name)

