import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU()):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.activation = activation

        self.hidden_layers = nn.ModuleList()
        last_size = input_size
        for hs in hidden_size:
            self.hidden_layers.append(nn.Linear(last_size, hs))
            last_size = hs

        self.output_layer = nn.Linear(last_size, output_size)

    def forward(self, x):
        # x = torch.flatten(x, 1)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        x = self.output_layer(x)
        return x


def train_mlp(model, train_loader, device, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")


def test_mlp(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y_batch.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f"Accuracy: {accuracy}")
    return accuracy
