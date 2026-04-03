
class VQVAEMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=2,
        latent_dim=2,
        encoder_layers=None,
        decoder_layers=None,
        vq_kwargs={},
        **kwargs,
    ):
        """Initializes the VQ-VAE model.

        Parameters
        ----------
        codebook_size : int, optional
            The size of the codebook. The default is 8.
        embed_dim : int, optional
            The dimension of the embedding space. The default is 2.
        input_dim : int, optional
            The dimension of the input data. The default is 2.
        encoder_layers : list, optional
            List of integers representing the number of units in each encoder layer.
            If None, a default encoder with a single linear layer is used. The default is None.
        decoder_layers : list, optional
            List of integers representing the number of units in each decoder layer.
            If None, a default decoder with a single linear layer is used. The default is None.
        """

        super().__init__()
        self.vq_kwargs = vq_kwargs
        self.embed_dim = latent_dim
        self.input_dim = input_dim  # for jet constituents, eta and phi

        # --- Encoder --- #
        if encoder_layers is None:
            self.encoder = torch.nn.Linear(self.input_dim, self.embed_dim)
        else:
            enc_layers = []
            enc_layers.append(torch.nn.Linear(self.input_dim, encoder_layers[0]))
            enc_layers.append(torch.nn.ReLU())

            for i in range(len(encoder_layers) - 1):
                enc_layers.append(torch.nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
                enc_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.Linear(encoder_layers[-1], self.embed_dim))

            self.encoder = torch.nn.Sequential(*enc_layers)

        # --- Vector-quantization layer --- #
        self.vqlayer = VectorQuant(feature_size=self.embed_dim, **vq_kwargs)

        # --- Decoder --- #
        if decoder_layers is None:
            self.decoder = torch.nn.Linear(self.embed_dim, self.input_dim)
        else:
            dec_layers = []
            dec_layers.append(torch.nn.Linear(self.embed_dim, decoder_layers[0]))
            dec_layers.append(torch.nn.ReLU())

            for i in range(len(decoder_layers) - 1):
                dec_layers.append(torch.nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
                dec_layers.append(torch.nn.ReLU())
            dec_layers.append(torch.nn.Linear(decoder_layers[-1], self.input_dim))

            self.decoder = torch.nn.Sequential(*dec_layers)

        self.loss_history = []
        self.lr_history = []

    def forward(self, samples, mask=None):
        # mask is there for compatibility with the transformer model
        # encode
        z_embed = self.encoder(samples)
        # quantize
        z_q2, vq_out = self.vqlayer(z_embed)
        # decode
        x_reco = self.decoder(z_q2)
        return x_reco, vq_out
