# Shakespeare Language Model

## Introduction
This project implements a transformer-based language model trained on the works of William Shakespeare. The model generates text based on the input context, allowing for creative writing and exploration of Shakespearean language.

## Features
- **Text Generation**: Generate new text based on the input context using a trained transformer model.
- **Customizable Hyperparameters**: Adjust training parameters such as batch size, learning rate, and number of epochs.
- **GPU Support**: Utilizes GPU for faster training if available.

## Tech Stack
- **Python**: The implementation is written in Python.
- **PyTorch**: A deep learning framework used for building and training the model.
- **Transformer Architecture**: Implements a transformer model for text generation.

## Installation
1. Clone the repository:
   ```bash
   git clone <YOUR_GIT_URL>
   cd shakespeare-llm
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install torch
   ```

## Usage
To train the model, run the following command:
```bash
python main.py
```
This will read the `shakespeare.txt` file, preprocess the text, and start training the model.

## Development
- The main implementation is in `main.py`. You can modify or extend the functionality as needed.
- Ensure to test your changes thoroughly to maintain the integrity of the model.

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes and create a pull request.

## License
This project is licensed under the MIT License. Feel free to use this project for personal or commercial purposes. 