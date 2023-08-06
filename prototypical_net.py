import torch.nn.functional as F
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_inputs: torch.Tensor,
        support_labels: torch.Tensor,
        query_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        support_inputs = {k:v.to(device) for k,v in support_inputs.items()}
        query_inputs = {k:v.to(device) for k,v in query_inputs.items()}

        z_support = self.backbone.base_model(**support_inputs).last_hidden_state[:, 0, :]
        z_query = self.backbone.base_model(**query_inputs).last_hidden_state[:, 0, :]

        # Normalize the embeddings
        z_support = F.normalize(z_support, p=2, dim=1)
        z_query = F.normalize(z_query, p=2, dim=1)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        # scores = torch.matmul(z_query, z_proto.t())

        return scores


