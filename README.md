# text-classification-with-Tensorflow

# 🎬 IMDb Sentiment Analysis using TensorFlow Hub

This project implements binary sentiment classification (positive/negative) on movie reviews from the IMDb dataset using a pre-trained text embedding from TensorFlow Hub.

---

## 📚 Dataset

The dataset used is the **IMDb Movie Reviews** dataset, provided by `tensorflow_datasets`.

- ✅ 25,000 labeled training examples
- ✅ 25,000 labeled test examples
- 🎯 Binary sentiment labels (`0 = negative`, `1 = positive`)

### Data Split:
- **Training**: 100% of the training set
- **Validation**: First 60% of test set
- **Testing**: Remaining 40% of test set

---

## 🔧 Project Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/imdb-sentiment-tfhub.git
cd imdb-sentiment-tfhub

2. Install dependencies

pip install -r requirements.txt
🔤 Text Embedding
We use a pre-trained text embedding layer from TensorFlow Hub:

🔗 Model: gnews-swivel-20dim

📏 Embedding Dimension: 20

📌 Trainable: Yes


hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
                           output_shape=[20], input_shape=[], dtype=tf.string, trainable=True)
🧠 Model Architecture

tf.keras.Sequential([
    hub_layer,                               # Pre-trained embedding
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])
Loss Function: BinaryCrossentropy(from_logits=True)

Optimizer: Adam

Metric: Accuracy

🚀 Training
The model is trained using:

🔁 20 epochs

🔀 Shuffling with buffer size 10000

🧪 Batch size of 100


history = model.fit(
    train_data.shuffle(10000).batch(100),
    epochs=20,
    validation_data=val_data.batch(100),
    verbose=1
)
📊 Evaluation
Model performance is evaluated on the test set:


results = model.evaluate(test_data.batch(100))
Output: Final test accuracy and loss values.

📦 Requirements
Create a requirements.txt with:

tensorflow
tensorflow-hub
tensorflow-datasets
numpy
Install with:


pip install -r requirements.txt

📬 Contact
For suggestions or improvements, feel free to fork the repo or open a pull request.

