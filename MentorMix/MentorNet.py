import torch
import torch.nn as nn
import torch.nn.functional as F

class MentorNet_arch(nn.Module):
    def __init__(self, label_embedding_size=2, epoch_embedding_size=5, num_label_embedding=2, num_fc_nodes=20):
        super(MentorNet_arch, self).__init__()
        self.feat_dim = label_embedding_size + epoch_embedding_size + 2  # plus 2 for bidirectional RNN output

        # Initialize embedding layers
        self.embedding_layer1 = nn.Embedding(num_embeddings=num_label_embedding, embedding_dim=label_embedding_size)
        self.embedding_layer2 = nn.Embedding(num_embeddings=101, embedding_dim=epoch_embedding_size)  # Including padding idx

        # Initialize RNN layer
        self.rnn = nn.RNN(input_size=2, hidden_size=1, bidirectional=True, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(self.feat_dim, num_fc_nodes)
        self.fc2 = nn.Linear(num_fc_nodes, 1)

    def forward(self, v_label, total_epoch, epoch, loss, loss_diff):
        label_embed = self.embedding_layer1(v_label)
        epoch = torch.full((loss.size(0),), epoch * 100 // total_epoch, dtype=torch.long, device=loss.device)
        epoch_embed = self.embedding_layer2(epoch)
        
        # Prepare loss data for RNN input
        rnn_input = torch.stack([loss, loss_diff], dim=-1)
        rnn_output, _ = self.rnn(rnn_input.unsqueeze(1))  # Shape (batch, seq_len=1, num_directions * hidden_size)
        rnn_output = rnn_output.squeeze(1)  # Remove the sequence dimension

        # Concatenate all features
        features = torch.cat([label_embed, epoch_embed, rnn_output], dim=1)

        # Fully connected layers
        x = F.tanh(self.fc1(features))
        x = torch.sigmoid(self.fc2(x)).squeeze()
        return x

# Example of initializing MentorNet_arch and using it in a training loop
def instantiate_and_use_mentornet():
    total_epochs = 150
    current_epoch = 10
    label = torch.tensor([1, 0, 1], dtype=torch.long)  # Example label tensor
    loss = torch.randn(3)  # Random loss values
    loss_diff = torch.randn(3)  # Random loss difference values

    mentor_net = MentorNet_arch()
    output = mentor_net(label, total_epochs, current_epoch, loss, loss_diff)
    print("Output from MentorNet:", output)

instantiate_and_use_mentornet()
