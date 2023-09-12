import torch
import gcn
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def samplegraph():

    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)

    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    print(data)

def data_set():
    datas = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    # datas = Planetoid(root='/tmp/Cora', name='Cora')
    print(datas[1])
    loader = DataLoader(datas, batch_size=32, shuffle=True)
    for batch in loader:
        print(batch)

        print(batch.num_graphs)

def transform():
    from torch_geometric.datasets import ShapeNet

    datas = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])
    print(datas[0])
    import torch_geometric.transforms as T
    from torch_geometric.datasets import ShapeNet

    datas = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                       pre_transform=T.KNNGraph(k=6))
    print(datas[0])

def train_model():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gcn.GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')


def main():
    # samplegraph()
    # data_set()
    # transform()
    train_model()

if __name__ == '__main__':
    main()

