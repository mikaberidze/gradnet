from gradnet import GradNet, fit
from gradnet.utils import plot_graph, laplacian, plot_adjacency_heatmap
from torch.linalg import eigvalsh


# define a loss function you want to minimize
def algebraic_connectivity(gn):
    # get the adjacency
    A = gn()
    L = laplacian(A)
    eigs = eigvalsh(L)
    return -eigs[1]


gn = GradNet(num_nodes=10, budget=10)
fit(gn=gn, loss_fn=algebraic_connectivity, num_updates=1000, accelerator="cuda")

plot_graph(gn, plt_show=True)
