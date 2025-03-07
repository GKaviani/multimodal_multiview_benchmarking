
import matplotlib.pyplot as plt
import Timeseries_model.tisc as tisc
from TS_Activity_Recognition.Dataset import TimeSeriesDataset
from TS_Activity_Recognition.data_util import *
import yaml
import argparse
from torch.utils.data import DataLoader

# Load config file
def load_config(config_file='/home/ghazal/Activity_Recognition_benchmarking/timeseries_unimodal/configs/imu_18d_config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Main function
def main(config_file=None):
    # Load configuration
    config = load_config(config_file) if config_file else load_config()

    # Dataset parameters
    train_dataset = TimeSeriesDataset(
        config['data']['train_data_path'],
        sampling_rate=config['data']['sampling_rate'],
        sequence_length=config['data']['sequence_length'],
        segments=config['data']['segments'],
        expected_dim=config['data']['expected_dim']
    )
    _, train_class_weights = create_weighted_sampler_and_class_weights(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], sampler=torch.utils.data.RandomSampler(train_dataset), drop_last=True) #shuffle=True,

    val_dataset = TimeSeriesDataset(
        config['data']['val_data_path'],
        sampling_rate=config['data']['sampling_rate'],
        sequence_length=config['data']['sequence_length'],
        segments=config['data']['segments'],
        expected_dim=config['data']['expected_dim']
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], drop_last=True)

    test_dataset = TimeSeriesDataset(
        config['data']['test_data_path'],
        sampling_rate=config['data']['sampling_rate'],
        sequence_length=config['data']['sequence_length'],
        segments=config['data']['segments'],
        expected_dim= config['data']['expected_dim']
    )
    test_dataloader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], drop_last=True)

    # Plot label distribution
    # plot_label_distribution(train_dataset, train_dataloader, config['output']['plot_filename'])

    # Model parameters
    IMU_classifier = tisc.build_classifier(
        model_name=config['model']['name'],
        timestep=(train_dataset.sampling_rate * train_dataset.sequence_length),
        dimentions=config['model']['dimensions'],
        num_layers=config['model']['num_layers'],
        num_classes=len(train_dataset.classes),
        output_base=config['model']['output_base'],
        output_name=config['model']['output_name'],
        class_weight=train_class_weights
    )

    # Train the model
    IMU_classifier.train(
        config['model']['num_epochs'],
        train_dataloader,
        val_dataloader,
        save_best_only=config['model']['save_best_only'],
        lr=config['model']['learning_rate']
    )

    # Evaluate the model
    IMU_classifier.evaluate(
        test_dataloader,
        return_report=True,
        return_confusion_matrix=True,
        with_best_model=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IMU Classifier with config.")
    parser.add_argument('--config', type=str, help='Path to the configuration file', default=None)
    args = parser.parse_args()

    main(config_file=args.config)

