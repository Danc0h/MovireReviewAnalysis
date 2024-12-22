import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load IMDB dataset
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data()

# Get word index
word_index = keras.datasets.imdb.get_word_index()

# Define word index and reverse word index
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# Function to decode review
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


'''# Pad sequences
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# Define and compile the model
model = keras.Sequential([
    keras.layers.Embedding(len(word_index), 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Split data for validation
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# Train the model
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# Evaluate on test data
results = model.evaluate(test_data, test_labels)
print("Test loss, Test accuracy:", results)

model.save("model.h5")'''
model = keras.models.load_model("model.h5")
with open("test.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded



'''Test a sample review
test_review = test_data[1]
# Reshape the test_review to match the model input shape
test_review = np.expand_dims(test_review, axis=0)
predict = model.predict(test_review)
print("Review:")
print(decode_review(test_data[0]))
print("Prediction:", str(predict[0][0]))
print("Actual Label:", str(test_labels[0]))'''
