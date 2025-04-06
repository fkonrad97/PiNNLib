from pinnLib.transformer.model.modular_transformer_model import ModularTransformerModel
from pinnLib.transformer.preprocessor.option_input_processor import OptionInputProcessor
from pinnLib.transformer.encoder.transformer_encoder import TransformerEncoder
from pinnLib.transformer.heads.option_price_head import OptionPriceHead

class OptionPricingTransformerModel(ModularTransformerModel):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        use_mlp: bool = False,
        use_positional: bool = True,
    ):
        input_processor = OptionInputProcessor(
            input_dim=input_dim,
            embed_dim=embed_dim,
            use_mlp=use_mlp,
            use_positional=use_positional,
        )

        encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        head = OptionPriceHead(embed_dim=embed_dim)

        super().__init__(input_processor=input_processor, encoder=encoder, head=head)