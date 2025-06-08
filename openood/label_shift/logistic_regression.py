import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SimpleDataset(Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        assert len(x) == len(y)
        self.data = x
        self.labels = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.linear.weight.data.fill_(1.0)
        self.linear.bias.data.fill_(0.01)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


def train_logistic_regression(data, labels, input_dim, output_dim):
    r"""
    Simple logistic regression model
    """
    print("Training a simple logistic regression model to rescale the output to [0,1]")
    dataset = SimpleDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    epochs = 100
    model = LogisticRegression(input_dim, output_dim)
    # defining the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # defining Cross-Entropy loss
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()

    Loss = []
    acc = []
    for epoch in range(epochs):
        correct = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x).squeeze()
            loss = criterion(outputs, y)
            # Loss.append(loss.item())
            loss.backward()
            optimizer.step()

            predicted = (outputs.detach().cpu() > 0.5)
            correct += (predicted == y).sum()

        lr_scheduler.step()
        Loss.append(loss.item())

        accuracy = 100 * correct / len(labels)
        acc.append(accuracy)
        if (epoch + 1) % 20 == 0:
            print('Logistic Regression Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch + 1, loss.item(), accuracy))
    params = list(model.parameters())

    print("Logsitic Regression training finished.")

    return float(params[0].detach().squeeze().cpu().numpy()), float(params[1].detach().squeeze().cpu().numpy())



if __name__ == '__main__':
    x = torch.rand((1000, 1))
    x[:500] += 1
    x[500:] -= 1
    y = torch.Tensor([1 for _ in range(500)] + [0 for _ in range(500)])
    print(train_logistic_regression(x, y, 1, 1))
