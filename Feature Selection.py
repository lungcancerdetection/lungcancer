import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Load and Preprocess Data ---
data = pd.read_csv('')

# Convert stringified lists into actual lists, then expand into numeric columns
for col in data.columns:
    if data[col].dtype == object and data[col].str.startswith('[').all():
        try:
            expanded = data[col].apply(ast.literal_eval)
            expanded_df = pd.DataFrame(expanded.tolist(), index=data.index)
            expanded_df.columns = [f"{col}_{i}" for i in range(expanded_df.shape[1])]
            data = pd.concat([data.drop(columns=[col]), expanded_df], axis=1)
        except Exception as e:
            print(f"Skipping column {col}: {e}")

# Separate features and label
X = data.drop(columns=['label', 'image'], errors='ignore')  # Drop non-feature columns
y = data['label']

# --- Feature Selection: Mutual Information Ranking ---
mi = mutual_info_classif(X, y, random_state=42)
mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)

# Plot Mutual Information Scores
top_n = 10
plt.figure(figsize=(12, 6))
mi_scores[:top_n].plot(kind='bar', color='skyblue')
plt.title(f"Top {top_n} Features by Mutual Information Score")
plt.ylabel("Mutual Information")
plt.xlabel("Features")
plt.xticks(rotation=30)
plt.grid(True)
plt.tight_layout()
plt.show()

# Retain top N features
X_selected = X[mi_scores.index[:top_n]]
X_np = X_selected.values
y_np = y.values

# --- Hybrid SMO + BCOA Optimizer Class ---
class HybridSMO_BCOA:
    def __init__(self, n_agents, max_iter, data, labels):
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.data = data
        self.labels = labels
        self.n_features = data.shape[1]
        self.population = np.random.randint(0, 2, (n_agents, self.n_features))
        self.best_agent = None
        self.best_fitness = -np.inf

    def fitness(self, agent):
        if np.sum(agent) == 0:
            return 0
        selected_features = self.data[:, agent == 1]
        X_train, X_test, y_train, y_test = train_test_split(
            selected_features, self.labels, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return acc

    def binary_transfer(self, x):
        return 1 / (1 + np.exp(-10 * (x - 0.5)))  # Sharper sigmoid

    def exploration_phase(self):
        for i in range(self.n_agents):
            r = np.random.randint(0, self.n_agents)
            diff = self.population[r] - self.population[i]
            move = np.random.rand(self.n_features) * diff
            transfer_move = np.where(self.binary_transfer(move) > np.random.rand(self.n_features), 1, 0)
            new_agent = np.clip(self.population[i] + transfer_move, 0, 1)
            if self.fitness(new_agent) > self.fitness(self.population[i]):
                self.population[i] = new_agent

    def exploitation_phase(self):
        for i in range(self.n_agents):
            r1, r2 = np.random.choice(self.n_agents, 2, replace=False)
            diff = self.population[r1] ^ self.population[r2]  # XOR operation
            rand_mask = np.random.randint(0, 2, size=self.n_features)
            move = diff * rand_mask
            new_agent = np.clip(self.population[i] ^ move, 0, 1)
            if self.fitness(new_agent) > self.fitness(self.population[i]):
                self.population[i] = new_agent

    def update(self):
        for iter in range(self.max_iter):
            if iter % 2 == 0:
                self.exploration_phase()
            else:
                self.exploitation_phase()

            for i in range(self.n_agents):
                fit = self.fitness(self.population[i])
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_agent = self.population[i].copy()

            if iter % 5 == 0 or iter == self.max_iter - 1:
                print(f"Iteration {iter+1}/{self.max_iter} -> Best Fitness: {self.best_fitness:.4f}")

    def get_selected_features(self, feature_names):
        return feature_names[self.best_agent == 1]

# --- Initialize and Run Optimizer ---
optimizer = HybridSMO_BCOA(
    n_agents=20,
    max_iter=50,
    data=X_np,
    labels=y_np
)

optimizer.update()

# --- Retrieve Selected Features and Save ---
selected_feature_names = optimizer.get_selected_features(X_selected.columns.values)
print("\nSelected Features after Hybrid SMO+BCOA:\n", selected_feature_names)

final_df = data[selected_feature_names.tolist() + ['label']]
output_path = ''
final_df.to_csv(output_path, index=False)
print(f"\nSelected features with corresponding values and labels saved to:\n{output_path}")
final_df