import torch
import pytest
from src.models.transformer import AraStudyTransformer, ModelArgs

@pytest.fixture
def model_args():
    return ModelArgs(
        dim=64,           # Small dimension for testing
        n_layers=2,
        n_heads=2,
        vocab_size=1000,
        max_seq_len=32,
        dropout=0.0
    )

def test_model_initialization(model_args):
    model = AraStudyTransformer(model_args)
    assert isinstance(model, AraStudyTransformer)
    assert model.args == model_args

def test_forward_pass_no_targets(model_args):
    model = AraStudyTransformer(model_args)
    batch_size = 2
    seq_len = 10
    tokens = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(tokens)
    
    assert logits.shape == (batch_size, seq_len, model_args.vocab_size)
    assert loss is None

def test_forward_pass_with_targets(model_args):
    model = AraStudyTransformer(model_args)
    batch_size = 2
    seq_len = 10
    tokens = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(tokens, targets)
    
    assert logits.shape == (batch_size, seq_len, model_args.vocab_size)
    assert loss is not None
    assert isinstance(loss.item(), float)

def test_mfu_estimation(model_args):
    model = AraStudyTransformer(model_args)
    mfu = model.estimate_mfu(fwdbwd_per_iter=10, dt=0.1)
    assert isinstance(mfu, float)

def test_gqa_forward_pass():
    model_args = ModelArgs(
        dim=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=1,  # MQA case
        vocab_size=1000,
        max_seq_len=32,
        dropout=0.0
    )
    model = AraStudyTransformer(model_args)
    batch_size = 2
    seq_len = 10
    tokens = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))
    
    logits, _ = model(tokens)
    assert logits.shape == (batch_size, seq_len, model_args.vocab_size)

