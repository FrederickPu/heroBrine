# heroBrine
A collection models for detecting anamolies transactions in bank data.
# heroBrine

## Overview

**heroBrine** is a project repository by FrederickPu, primarily developed using Jupyter Notebook. The repository is public and open for collaboration, forking, and issue tracking.

> **Note:** The project description has not yet been provided. If you are the author or a contributor, please add a more detailed project description, key features, and usage instructions below.

## Features

- Written in Jupyter Notebook format for interactive code execution and documentation.
- Open-source and available for forking and contributions.
- Supports collaborative project management with GitHub issues and wiki.

## Getting Started

To get started with this project:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/FrederickPu/heroBrine.git
    cd heroBrine
    ```

2. **Open Jupyter Notebooks:**
    - Launch Jupyter Notebook or JupyterLab in your project directory.
    - Open the relevant notebook files to explore or run the code.


> _Feel free to update this README with more specific project details, features, and usage instructions as the project evolves._

## running pretrained checkpoints

model weights for the lstm model can be found in `model_lstm.h5` and the bidirectional weights can be found in `model_lstm_bi`. These checkpoints can be loaded using `tf.keras.models.load_model`

## running the benchmarking notebook

When running the `bencmarking.ipynb` you may need to change the file paths for the csv files namely:
```
kyc = pd.read_csv('/Users/mac/Desktop/kyc.csv')
emt = pd.read_csv('/Users/mac/Desktop/emt.csv')
wire = pd.read_csv('/Users/mac/Desktop/wire.csv')
abm = pd.read_csv('/Users/mac/Desktop/abm.csv')
cheque = pd.read_csv('/Users/mac/Desktop/cheque.csv')
card = pd.read_csv('/Users/mac/Desktop/card.csv')
eft = pd.read_csv('/Users/mac/Desktop/eft.csv')
```

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

## License

_The repository currently does not specify a license. Please contact the author for usage and distribution permissions._

## Author

[FrederickPu](https://github.com/FrederickPu)

---
