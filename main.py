import torch
from torch_geometric.data import Data

def samplegraph():
    # 边，shape = [2,num_edge]
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    # 点，shape = [num_nodes, num_node_features]
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    print(data)
    # Data(edge_index=[2, 4], x=[3, 1])


def main():
    samplegraph()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
