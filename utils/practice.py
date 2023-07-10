import torch


class MegaByteDecoder:
    def __init__(
            self,
            global_args,
            local_args,
            patch_size,
    ):
        self.pad = 0
        self.patch_size = patch_size
        self.global_model = TransformerDecoder(global_args)
        self.localmodel = TransformerDecoder(local_args)

    def forward(self,
                bytes):
        bytes_global, bytes_local = self.prepare_input(bytes)

        global_bytes_embedded = self.globalmodel.embed(bytes_global)
        global_in = rearrange(
            global_bytes_embedded,
            "b (t p) e -> b t (p e)",
            p = self.patch_size,
        )
        global_output = self.globalmodel(global_in)

        global_output_reshaped = rearrange(
            global_output,
            "b t (p e) -> (b t) p e",
            p=self.patch_size,
        )
        local_bytes_embedded = self.localmodel.embed(bytes_local)
        local_in = local_bytes_embedded + global_output_reshaped
        local_output = self.localmodel(local_in)

        batch_size = bytes_global.shape[0]
        x = rearrange(local_output, "(b t) 1 v -> b (t 1) v", b=batch_size)
        return x
    
    def prepare_input(self, bytes):
        padding_global = bytes.new(bytes.shape[0], self.patch_size).fill_(self.pad)
        bytes_global = torch.cat((padding_global, bytes[:, :, -self.patch_size]), -1)

        bytes_input = rearrange(bytes, "b (t p) -> (b t) p", p=self.patch_size)
        padding_local = bytes_input.new(bytes_input.shape[0], 1).fill_(self.pad)
        bytes_local = torch.cat((padding_local, bytes_input[:, :-1]), -1)

        return bytes_global, bytes_local
    
    