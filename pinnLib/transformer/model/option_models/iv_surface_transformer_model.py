from torch import nn
from pinnLib.transformer.model.modular_transformer_model import ModularTransformerModel
from pinnLib.transformer.preprocessor.option_input_processor import OptionInputProcessor
from pinnLib.transformer.encoder.transformer_encoder import TransformerEncoder
from pinnLib.transformer.heads.option_vol_surface_head import VolSurfaceHead

class IVSurfaceTransformerModel(ModularTransformerModel):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_layers: int,
        num_strikes: int,
        num_maturities: int,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
        use_mlp: bool = False,
        use_positional: bool = True,
    ):
        input_processor = OptionInputProcessor(
            input_dim=input_dim,  # should be 2 if just K, T
            embed_dim=embed_dim,
            use_mlp=use_mlp,
            use_positional=use_positional
        )

        encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        head = VolSurfaceHead(
            embed_dim=embed_dim,
            hidden_dim=head_hidden_dim,
            num_strikes=num_strikes,
            num_maturities=num_maturities
        )

        super().__init__(input_processor=input_processor, encoder=encoder, head=head)
