import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

STUDENT_ID = "20244022"
MAX_LENGTH = 20

def p1_load_data():
    
    train_path = './simple_seq.train.csv'
    test_path = './simple_seq.test.csv'
    
    def one_hot_encode(token, vocab_size):
        vec = np.zeros(vocab_size, dtype=int)
        if token in W_token_to_id:
            vec[W_token_to_id[token]] = 1
        return vec
    
    with open(train_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().rstrip(',') for line in f.readlines()]
        
    token_sequences = [line.split(',') for line in lines]
    
    unique_tokens = sorted(set(token for seq in token_sequences for token in seq))
    W_tokens = [token for token in unique_tokens if token.startswith('W')]
    D_tokens = [token for token in unique_tokens if token.startswith('D')]
    W_token_to_id = {token: idx+1 for idx, token in enumerate(W_tokens)}
    W_token_to_id['PAD'] = 0 
    D_token_to_id = {token: idx for idx, token in enumerate(D_tokens)}
    
    encoded_sequences = []
    test_encoded_sequences = []
    labels = []
    vocab_size = len(W_tokens)+1
    
    for data in token_sequences:
        train_data = [token for token in data if token in W_token_to_id]
        label = data[-1]
        
        for _ in range(MAX_LENGTH-len(train_data)):
            train_data.append('PAD')
        one_vec=[]
        
        for i in train_data:
            one_vec.append(one_hot_encode(i, vocab_size))
        encoded_sequences.append(one_vec)
        labels.append(D_token_to_id.get(label, 0))
        
    train_data = torch.tensor(encoded_sequences, dtype=torch.float32)
    train_label = torch.tensor(labels, dtype=torch.long)
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_lines = [line.strip().rstrip(',') for line in f.readlines()]
        
    test_token_sequences = [line.split(',') for line in test_lines]
    
    for data in test_token_sequences:
        test_data = [token for token in data if token in W_token_to_id]
        
        for _ in range(MAX_LENGTH-len(test_data)):
            test_data.append('PAD')
        test_one_vec=[]
        
        for i in test_data:
            test_one_vec.append(one_hot_encode(i, vocab_size))
        test_encoded_sequences.append(test_one_vec)
        
    test_data = torch.tensor(test_encoded_sequences, dtype=torch.float32)
    
    return train_data, train_label, test_data, D_token_to_id, test_token_sequences

def p1_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, train_label, test_data, D_token_to_id, test_token_sequences = p1_load_data()
    print(f'{train_data.shape},' f'{train_label.shape}')
    
                
    class Linear(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.w = nn.Parameter(torch.empty(input_dim, output_dim))
            torch.nn.init.xavier_normal_(self.w)
            self.b = nn.Parameter(torch.zeros(output_dim))
            
        def forward(self, x):
            output = torch.matmul(x, self.w)+self.b
            return output
        
    class multilayer(nn.Module):
        def __init__(self, input_dim, hidden_dim, divide_factor, output_dim):
            super().__init__()
            self.layer1 = Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(0.5)
            self.layer2 = Linear(hidden_dim, hidden_dim//divide_factor)
            self.bn2 = nn.BatchNorm1d(hidden_dim//divide_factor)
            self.layer3 = Linear(hidden_dim//divide_factor, output_dim)
            self.activation = nn.Sigmoid()
                
        def forward(self, x):
            batch_size, seq_length, vocab_size = x.shape
            x = x.view(batch_size, -1)
            o1=self.activation(self.bn1(self.layer1(x)))
            o1=self.dropout(o1)
            o2=self.activation(self.bn2(self.layer2(o1)))
            o2=self.dropout(o2)
            o3=self.layer3(o2)
            return o3
        
        
    def evaluate(model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                predictions = torch.argmax(logits, dim=1)

                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total * 100
        return accuracy
    
    batch_size = 30
    vocab_size = train_data.shape[2] #2547
    input_dim = vocab_size * MAX_LENGTH #2547*20
    print(input_dim)
    hidden_dim = 200
    output_dim = len(train_label.unique())
    print(output_dim)
    divide_factor = 2
    
    model = multilayer(input_dim, hidden_dim, divide_factor, output_dim).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_label, test_size=0.2, random_state=42
    )
    
    train_data, train_labels = train_data, train_labels
    valid_data, valid_labels = valid_data, valid_labels
    
    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valid_data, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 50
    best_valid_accuracy = 0
    best_model_state = None

    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        valid_accuracy = evaluate(model, valid_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_model_state = model.state_dict()

    print("Training complete!")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_accuracy = evaluate(model, valid_loader)
    print(f"Final Validation Accuracy: {final_accuracy:.2f}%")
    
    
    model.eval()
    with torch.no_grad():
        test_logits = model(test_data.to(device))
        test_predictions = torch.argmax(test_logits, dim=1)
    
    id_to_D_token = {idx: token for token, idx in D_token_to_id.items()}
    predicted_tokens = [id_to_D_token[pred.item()] for pred in test_predictions]
    
    ids = [f"S{str(i+1).zfill(3)}" for i in range(len(predicted_tokens))]
    
    df_test = pd.DataFrame({
        'id': ids,
        'pred': predicted_tokens
    })

    df_test.to_csv(f'{STUDENT_ID}_simple_seq.p1.answer.csv', index=False)

    print("Test predictions saved!")

def p2_load_data():
    
    train_path = './simple_seq.train.csv'
    test_path = './simple_seq.test.csv'
    
    def one_hot_encode(token, vocab_size):
        vec = np.zeros(vocab_size, dtype=int)
        if token in W_token_to_id:
            vec[W_token_to_id[token]] = 1
        return vec
    
    with open(train_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().rstrip(',') for line in f.readlines()]
        
    token_sequences = [line.split(',') for line in lines]
    
    unique_tokens = sorted(set(token for seq in token_sequences for token in seq))
    W_tokens = [token for token in unique_tokens if token.startswith('W')]
    D_tokens = [token for token in unique_tokens if token.startswith('D')]
    W_token_to_id = {token: idx+1 for idx, token in enumerate(W_tokens)}
    W_token_to_id['PAD'] = 0 
    D_token_to_id = {token: idx for idx, token in enumerate(D_tokens)}
    
    encoded_sequences = []
    test_encoded_sequences = []
    labels = []
    vocab_size = len(W_tokens)+1
    
    for data in token_sequences:
        train_data = [token for token in data if token in W_token_to_id]
        label = data[-1]
        
        for _ in range(MAX_LENGTH-len(train_data)):
            train_data.append('PAD')
        one_vec=[]
        
        for i in train_data:
            one_vec.append(one_hot_encode(i, vocab_size))
        encoded_sequences.append(one_vec)
        labels.append(D_token_to_id.get(label, 0))
        
    train_data = torch.tensor(encoded_sequences, dtype=torch.float32)
    train_label = torch.tensor(labels, dtype=torch.long)
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_lines = [line.strip().rstrip(',') for line in f.readlines()]
        
    test_token_sequences = [line.split(',') for line in test_lines]
    
    for data in test_token_sequences:
        test_data = [token for token in data if token in W_token_to_id]
        
        for _ in range(MAX_LENGTH-len(test_data)):
            test_data.append('PAD')
        test_one_vec=[]
        
        for i in test_data:
            test_one_vec.append(one_hot_encode(i, vocab_size))
        test_encoded_sequences.append(test_one_vec)
        
    test_data = torch.tensor(test_encoded_sequences, dtype=torch.float32)
    
    return train_data, train_label, test_data, D_token_to_id, test_token_sequences

    
def p2_main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, train_label, test_data, D_token_to_id, test_token_sequences = p2_load_data()
    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_label.shape}")
    
    vocab_size = train_data.shape[2]
    embed_dim = 64
    max_length = MAX_LENGTH
    hidden_dim = 200
    output_dim = len(train_label.unique())
    divide_factor = 2
    batch_size = 30
    num_epochs = 50
    best_valid_accuracy = 0
    best_model_state = None
    
    class TrainableDictionary(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.dictionary = nn.Parameter(torch.empty(vocab_size, embed_dim))
            torch.nn.init.xavier_normal_(self.dictionary)

        def forward(self, one_hot):
            embedded = torch.matmul(one_hot, self.dictionary)
            embedded_normalized = nn.functional.normalize(embedded, p=2, dim=-1)
            return embedded_normalized

    class Linear(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.w = nn.Parameter(torch.empty(input_dim, output_dim))
            torch.nn.init.xavier_normal_(self.w)
            self.b = nn.Parameter(torch.zeros(output_dim))
            
        def forward(self, x):
            output = torch.matmul(x, self.w)+self.b
            return output
        
    class multilayer(nn.Module):
        def __init__(self, vocab_size, embed_dim, max_length, hidden_dim, divide_factor, output_dim):
            super().__init__()
            self.embedding = TrainableDictionary(vocab_size, embed_dim)
            self.layer1 = Linear(embed_dim*max_length, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(0.5)
            self.layer2 = Linear(hidden_dim, hidden_dim//divide_factor)
            self.bn2 = nn.BatchNorm1d(hidden_dim//divide_factor)
            self.layer3 = Linear(hidden_dim//divide_factor, output_dim)
            self.activation = nn.Sigmoid()
                
        def forward(self, x):
            x_embed = self.embedding(x)
            batch_size, seq_length, vocab_size = x_embed.shape
            x_embed = x_embed.view(batch_size, -1)
            o1=self.activation(self.bn1(self.layer1(x_embed)))
            o1=self.dropout(o1)
            o2=self.activation(self.bn2(self.layer2(o1)))
            o2=self.dropout(o2)
            o3=self.layer3(o2)
            return o3
        
    model = multilayer(vocab_size, embed_dim, max_length, hidden_dim, divide_factor, output_dim).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_label, test_size=0.2, random_state=42
    )
    
    train_data, train_labels = train_data, train_labels
    valid_data, valid_labels = valid_data, valid_labels
    
    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valid_data, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    def evaluate(model, dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
        return correct / total * 100
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        valid_accuracy = evaluate(model, valid_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_model_state = model.state_dict()
    
    print("Training complete!")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_accuracy = evaluate(model, valid_loader)
    print(f"Final Validation Accuracy: {final_accuracy:.2f}%")
    
    model.eval()
    with torch.no_grad():
        test_logits = model(test_data.to(device))
        test_predictions = torch.argmax(test_logits, dim=1)
    
    id_to_D_token = {idx: token for token, idx in D_token_to_id.items()}
    predicted_tokens = [id_to_D_token[pred.item()] for pred in test_predictions]
    ids = [f"S{str(i+1).zfill(3)}" for i in range(len(predicted_tokens))]
    df_submit = pd.DataFrame({
        'id': ids,
        'pred': predicted_tokens
    })
    df_submit.to_csv(f'{STUDENT_ID}_simple_seq.p2.answer.csv', index=False)
    print("Test predictions saved!")
    
if __name__ == "__main__":
    p1_main()
    p2_main()
