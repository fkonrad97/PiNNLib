import torch.nn as nn

class ModularTransformerModel(nn.Module):
    def __init__(self, input_processor, encoder, head):
        super().__init__()
        self.input_processor = input_processor
        self.encoder = encoder
        self.head = head

    def forward(self, x, mask=None):
        x = self.input_processor(x)
        x = self.encoder(x, mask)
        return self.head(x)