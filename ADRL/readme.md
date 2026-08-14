# RL-ICSA: A Reinforcement Learning Framework for Adaptive Collusion Byzantine Attacks with Inter-Class Similarity Action

## Project Structure

```text
ADRL/
├── actors.py              # Client and server Actor classes
├── config.py              # Dataset configurations and transformations
├── main.py                # Main execution file
├── strategies.py          # Federated learning strategies
├── utilities.py           # Utility functions
├── requirements.txt       # Required dependencies
├── README.md              # Project documentation
├── models/                # Model definitions
│   ├── __init__.py
│   └── cifar10.py         # Simple CNN model
├── byzantine/             # Byzantine attacks and defenses
│   ├── __init__.py
│   ├── attacks            # Attack methods
│   ├── defences           # Defense methods
│   └── config.py          # Configuration
├── runners/               # Runners
│   ├── __init__.py
│   ├── BaseRunner.py      # Base runner
│   └── FedAVGRunner.py    # FedAVG runner
```

# Features

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --dataset cifar10 --arch resnet18 --n_clients 50 --n_communications 30 --n_total_local_epochs 150 --attack poisonedfl --defence momentum --n_byzantine 10
```

### Parameter Description

- `--dataset`: Dataset name (`mnist`, `cifar10`, `cifar100`, `fashionmnist`)
- `--arch`: Model architecture (`simple_cnn`, `mnist_cnn`, `simple_resnet`)
- `--n_clients`: Number of clients
- `--n_communications`: Number of communication rounds
- `--n_total_local_epochs`: Total number of local training epochs
- `--batch_size`: Batch size

## Configuration

The project supports configuring training parameters either through command-line arguments or by modifying the `defaults` dictionary in `main.py`.

### Main Configuration Options

- **Dataset settings**: `dataset`, `batch_size`
- **Model settings**: `arch`, `input_channels`
- **Federated learning settings**: `n_clients`, `n_communications` (FedAVG is used by default)
- **Byzantine settings**: `attack`, `defence`, `n_byzantine_clients`
- **Training settings**: `lr`, `optimizer`, `scheduler`
- **System settings**: `device`, `seed`, `use_amp`

## Output Results

Training results will be saved in the `./results/` directory, including:

- `config.json`: Training configuration
- `final_results.json`: Final results and metrics
- `final_model.pth`: Trained model

## Extending the Project

### Adding a New Model

Create a new model file under the `models/` directory and add the corresponding model to the `get_model` function in `models/cifar10.py`.

### Modifying the Strategy

The project uses FedAVG as the default strategy. To modify it, adjust the implementation of the FedAVG class in `strategies.py`.

### Adding a New Dataset

Add the configuration for the new dataset in `config.py`, including its mean, standard deviation, and data transformations.

### Byzantine Attacks and Defenses

The project supports multiple Byzantine attack and defense methods.

#### Attack Methods

- `no_attack`: No attack (baseline)
- `ICSA`: Inter-Class Similarity Attack
- `qlearning`: Q-Learning

#### Defense Methods

- `no_defence`: No defense (baseline)
- `median`: Median defense
- `trimmed_mean`: Trimmed Mean defense
- `krum`: Krum defense
- `bulyan`: Bulyan defense
- `fltrust`: FLTrust defense
- `fool`: FoolsGold defense
- `ada`: AdaAggRL defense
- `momentum`: Momentum defense

## Notes

1. Make sure sufficient GPU memory is available for training.
2. Adjust the batch size according to the dataset size.
3. The performance of federated learning depends on the number of clients and communication rounds.
4. Small-scale tests are recommended before running large-scale experiments.

## License

This project is released under the MIT License.
