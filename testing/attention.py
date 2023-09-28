import unittest
import torch
from torch.nn import Dropout

from torch import einsum
from torch import nn
from torch.testing import assert_allclose

from terra_byte.model.attend import Attend, FlashAttention, EfficientAttentionConfig


class TestAttending(unittest.TestCase):
    def setUp(self):
        self.attend = Attend(dim=512)

    def test_init_default(self):
        self.assertEqual(self.attend_dim, 512)
        self.assertEqual(self.attend.heads, 64)
        self.assertEqual(self.attend.dim_head, 64)