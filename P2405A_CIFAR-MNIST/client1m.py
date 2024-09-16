# %%
import flwr as fl
import tensorflow as tf

# NC_BEG

# Carregar o conjunto de dados MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os valores de pixel para o intervalo [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Camada de entrada: achatamento da imagem
    tf.keras.layers.Dense(128, activation='relu'),  # Primeira camada oculta com 128 neurônios e função de ativação ReLU
    tf.keras.layers.Dropout(0.2),  # Regularização com dropout de 20%
    tf.keras.layers.Dense(10, activation='softmax')  # Camada de saída com 10 neurônios para as 10 classes e função de ativação softmax
])


# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Função de perda para problemas de classificação
              metrics=['accuracy'])


# NC_END



# Define Flower client
class MnistClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return model.get_weights()

  def fit(self, parameters, config):
    model.set_weights(parameters)
    model.fit(x_train, y_train, epochs=1, batch_size=32)
    return model.get_weights(), len(x_train), {}

  def evaluate(self, parameters, config):
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, len(x_test), {"accuracy": accuracy}



# %% [markdown]
# Starting Flower Client

# %%
fl.client.start_client(
		server_address='127.0.0.1:8080',
		client=MnistClient().to_client(), # <-- where FlowerClient is of type flwr.client.NumPyClient object
	)

# %%



