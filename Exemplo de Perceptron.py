import numpy as np

class Perceptron:
    """
    Implementação do classificador Perceptron de Rosenblatt.
    ... (docstrings) ...
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Ajusta o modelo aos dados de treinamento.
        ... (docstrings) ...
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            
            self.errors_.append(errors)
            
            # --- Parada Antecipada ---
            if errors == 0:
                print(f"Convergência alcançada na época {i+1}.")
                return self

        print(f"Máximo de épocas ({self.n_iter}) atingido sem convergência completa.")
        return self

    def net_input(self, X):
        """Calcula a entrada líquida (soma ponderada)"""
        X_proc = np.atleast_1d(X)
        return np.dot(X_proc, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Retorna o rótulo da classe (1 ou -1) após a função degrau"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# --- Exemplo de Uso (Problema XOR, não linearmente separável) ---
# O Perceptron NÃO deve convergir aqui
print("--- Teste do Perceptron (Porta Lógica XOR) ---")
X_xor_perc = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor_perc = np.array([-1, 1, 1, -1]) # Rótulos -1 e 1 para XOR

ppn_xor = Perceptron(eta=0.1, n_iter=100)
ppn_xor.fit(X_xor_perc, y_xor_perc)
print(f"Erros por época (XOR): {ppn_xor.errors_}") # Erros não chegam a zero
print(f"Predições (XOR): {ppn_xor.predict(X_xor_perc)}")
# Acurácia de 50% ou 75% (falha)
