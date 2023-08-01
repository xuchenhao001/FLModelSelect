import torch
import torch.nn.functional as F


def test_img(net_g, dataset, test_indices, local_test_bs, device):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = dataset.load_test_dataset(test_indices, local_test_bs)
    for idx, (data, target) in enumerate(data_loader):
        data = data.detach().clone().type(torch.FloatTensor)
        if device != torch.device('cpu'):
            data, target = data.to(device), target.to(device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum')
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss
