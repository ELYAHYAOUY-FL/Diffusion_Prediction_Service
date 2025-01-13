from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import networkx as nx
import matplotlib.pyplot as plt
import random
import logging

# Créer l'application Flask
app = Flask(_name_)

CORS(app)
# Load the first 1000 rows of the dataset
data_path = 'data/datasplit_part1.csv'
df = pd.read_csv(data_path, nrows=5000)

# Initial data overview
print("Initial Dataset Shape:", df.shape)
print("Columns with Missing Values:\n", df.isnull().sum())

# Impute numerical columns with median and categorical columns with the most frequent value
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df = remove_outliers(df, numerical_cols)

# Créer un ensemble de toutes les IPs (source et destination)
all_ips = set(df['src_ip'].unique()) | set(df['dst_ip'].unique())

# Préparer l'encodeur avec toutes les IPs possibles
encoder = LabelEncoder()
encoder.fit(list(all_ips))

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

print("Preprocessed Dataset Shape:", df.shape)

G = nx.Graph()

# Add edges based on communication data
for _, row in df.iterrows():
    src_ip = row['src_ip']
    dst_ip = row['dst_ip']
    weight = row['pktTotalCount']  # Edge weight

    if not G.has_edge(src_ip, dst_ip):
        G.add_edge(src_ip, dst_ip, weight=weight)
    else:
        G[src_ip][dst_ip]['weight'] += weight

print(f"Graph constructed with {len(G.nodes)} nodes and {len(G.edges)} edges.")

# Visualize a subset of the graph
plt.figure(figsize=(12, 8))
subgraph = nx.subgraph(G, list(G.nodes)[:1000])  # Visualize a subset of the graph
pos = nx.spring_layout(subgraph)
nx.draw(subgraph, pos, with_labels=True, node_size=500, font_size=8)
plt.title("Sample Graph Visualization")
path = 'static/graph.png'
plt.savefig(path)

def simulate_diffusion(G, initial_infected, infection_prob, steps=10):
    infection_states = {node: 0 for node in G.nodes}
    for node in initial_infected:
        infection_states[node] = 1

    infection_history = [infection_states.copy()]
    for _ in range(steps):
        new_states = infection_states.copy()
        for node in G.nodes:
            if infection_states[node] == 1:
                for neighbor in G.neighbors(node):
                    if infection_states[neighbor] == 0:
                        if random.random() < infection_prob:
                            new_states[neighbor] = 1
        infection_states = new_states
        infection_history.append(infection_states.copy())

    return infection_history

# Simulate infection spread
initial_infected = [random.choice(list(G.nodes))]
infection_prob = 0.3
infection_history = simulate_diffusion(G, initial_infected, infection_prob, steps=10)

def prepare_training_data(G, infection_history):
    X = []
    y = []

    for t in range(1, len(infection_history)):
        current_state = infection_history[t - 1]
        next_state = infection_history[t]

        for node in G.nodes:
            features = [
                G.degree[node],
                current_state[node],
                np.mean([current_state[neighbor] for neighbor in G.neighbors(node)]),
            ]
            X.append(features)
            y.append(next_state[node])

    return np.array(X), np.array(y)

# Prepare training data
X, y = prepare_training_data(G, infection_history)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def extract_features(G):
    features = []
    for node in G.nodes:
        degree = G.degree[node]
        betweenness = nx.betweenness_centrality(G).get(node, 0)
        closeness = nx.closeness_centrality(G).get(node, 0)
        features.append([degree, betweenness, closeness])
    return np.array(features)

def test_ip_impact(G, model, test_ip, infection_prob=0.3, steps=10):
    if test_ip not in G.nodes:
        print(f"IP {test_ip} not found in the graph.")
        return None

    initial_infected = [test_ip]
    infection_history = simulate_diffusion(G, initial_infected, infection_prob, steps)
    final_infection_states = infection_history[-1]
    features = extract_features(G)
    predictions = model.predict(features)
    results = {node: final_infection_states[node] for node in G.nodes}
    return results

def visualize_impact(G, test_ip, encoded_ip, impact_results):
    affected_count = sum(1 for node in G.nodes if impact_results.get(node, 0) == 1)
    unaffected_count = len(G.nodes) - affected_count

    node_colors = [
        "red" if node == encoded_ip else
        ("orange" if impact_results.get(node, 0) == 1 else "blue")
        for node in G.nodes
    ]

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=50,
        font_size=8
    )

    red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Test IP', markerfacecolor='red', markersize=10)
    orange_patch = plt.Line2D([0], [0], marker='o', color='w', label='Affected Nodes', markerfacecolor='orange', markersize=10)
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Unaffected Nodes', markerfacecolor='blue', markersize=10)
    plt.legend(handles=[red_patch, orange_patch, blue_patch], loc="upper right")

    plt.title(f"Impact of IP {test_ip} on the Network\n"
              f"Affected Nodes: {affected_count}, Unaffected Nodes: {unaffected_count}")

    path2 = 'static/graphaffected.png'
    plt.savefig(path2)
    plt.close()

@app.route('/test_ip', methods=['GET', 'POST'])
def test_ip():
    print("IPs disponibles:", list(all_ips)[:10])  # Debug: Voir les IPs valides
    
    if request.method == 'POST':
        try:
            # Get the IP address from the request body (from Angular)
            data = request.get_json()  # Expecting JSON payload from Angular
            test_ip = data.get('ip_address')
            print(f"Chosen IP: {test_ip}")
            
            encoded_ip = encoder.transform([test_ip])[0]  # Assuming encoder was previously fitted
            print(f"Encoded IP: {encoded_ip}")

            impact_results = test_ip_impact(G, clf, encoded_ip)
            print(f"Impact Results: {impact_results}")

            if impact_results:
                visualize_impact(G, test_ip, encoded_ip, impact_results)
                # Return a success message along with the path to the updated image
                return jsonify({
                    'status': 'success',
                    'message': 'Impact results generated successfully.',
                    'graph_path': '/static/graphaffected.png'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Error: Could not generate impact results'
                })
                
        except Exception as e:
            error_message = f"Error processing IP address: {str(e)}"
            return jsonify({
                'status': 'error',
                'message': error_message
            })
    
    # If it's a GET request, return the page (for Angular to load initially)
    return render_template('index.html')

from flask import Flask, jsonify, request

if _name_ == '_main_':
    app.run(debug=True, port=5001)  # Changez 5001 par un port non utilisé
