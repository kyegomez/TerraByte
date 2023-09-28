from terra_byte.model.model import TerraByte
import torch


class TerraByte:
    def __init__(self,
                num_tokens = 16000,
                dim = (512, 256),   
                dilation_rate=4, 
                segment_size=2,
                max_seq_len = (1024, 4), 
                depth = (6, 4),          
                dim_head = 64,           
                heads = 8,  
                 ):
        self.model = TerraByte(
            num_tokens=num_tokens,
            dim=dim,
            max_seq_len=max_seq_len,
            dilation_rate=dilation_rate,
            segment_size=segment_size,
            depth=depth,
            dim_head=dim_head,
            heads=heads
        )

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
        return logits
    
    def generate(
            self,
            num_tokens=1024,
            temperature=0.9,
            filter_thres=0.9
        ):
        self.model.eval()
        with torch.no_grad():
            sampled = self.model.generate(num_tokens, temperature=temperature, filter_thres=filter_thres)
        return sampled
    
    
