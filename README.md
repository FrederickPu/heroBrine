# heroBrine
A collection models for detecting anamolies transactions in bank data.

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
