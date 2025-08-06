import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import yaml

from core.mycelial_model import MycelialSpiralformer
from utils.breath_clock import BreathClock
from utils.rhythmic_loss import RhythmicLossWrapper
from data.mycelial_datagen import MycelialDataGenerator, NetworkConditions

# --- Constants & Config ---
CONFIG_PATH = Path(__file__).resolve().parents[2] / "spiralformer_parameters.yml"

def load_config(config_path: Path) -> dict:
    """Loads the YAML configuration for the experiment."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class MycelialDataset(Dataset):
    """A PyTorch dataset for the mycelial data."""
    def __init__(self, data_path: str, config: dict):
        self.samples = []
        self.config = config
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        cfg_model = self.config['models']['femto_mycelial_cpu'] # or whichever is used
        cfg_shared = self.config['shared']['contemplative']

        conditions = NetworkConditions(**sample['conditions'])
        condition_vec = torch.tensor(conditions.to_condition_vector(), dtype=torch.float32)

        glyph_seq = sample['glyph_sequence']
        # Simple padding/truncating
        glyph_seq = glyph_seq[:cfg_model['seq_len']-1] + [0] * (cfg_model['seq_len'] - 1 - len(glyph_seq))
        
        input_tokens = torch.tensor([cfg_shared['silence_id']] + glyph_seq, dtype=torch.long)
        target_tokens = torch.tensor(glyph_seq + [cfg_shared['silence_id']], dtype=torch.long)

        return input_tokens, target_tokens, condition_vec

def main(config_name: str = "femto_mycelial_cpu"):
    """The main training loop for the MycelialSpiralformer."""
    config = load_config(CONFIG_PATH)
    
    if config_name not in config['models']:
        raise ValueError(f"Configuration '{config_name}' not found in parameters file.")
    
    cfg_model = config['models'][config_name]
    cfg_shared = config['shared']['contemplative']
    cfg_training = cfg_model['training']
    cfg_data = cfg_model['data']

    print(f"🍄 Training Mycelial Spiralformer using '{config_name}' configuration...")
    
    # --- Ensure model directory exists ---
    model_dir = Path(cfg_model['save_paths']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"  -> Models will be saved in: {model_dir}")

    # 1. Generate Data
    print("  -> Step 1: Generating mycelial training data...")
    datagen = MycelialDataGenerator(random_seed=42)
    data_path = datagen.generate_training_dataset(
        num_echoes=cfg_data['num_samples'], 
        chaos_mode=cfg_data['chaos_mode']
    )
    print(f"  -> Data generation complete. Data saved to {data_path}")

    dataset = MycelialDataset(data_path, config)
    # Using num_workers=0 is crucial for stability on many systems, especially laptops.
    dataloader = DataLoader(dataset, batch_size=cfg_training['batch_size'], shuffle=True, num_workers=0)
    print("  -> Step 2: Dataset and DataLoader initialized.")


    # 2. Initialize Model and Components
    device = torch.device(cfg_model.get("target_device", "cpu"))
    print(f"Using device: {device}")

    clock = BreathClock(**cfg_shared['breath_clock'])
    model = MycelialSpiralformer(
        vocab_size=cfg_shared['vocab_size'],
        d_model=cfg_model['d_model'],
        n_heads=cfg_model['n_heads'],
        num_layers=cfg_model['num_layers'],
        seq_len=cfg_model['seq_len'],
        condition_dim=cfg_model['condition_dim']
    ).to(device)
    criterion = RhythmicLossWrapper(nn.CrossEntropyLoss(ignore_index=0), clock) # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_training['learning_rate'])

    # 3. Training Loop
    print("  -> Step 3: Beginning training loop...")
    for epoch in range(cfg_training['epochs']):
        total_loss = 0
        for inputs, targets, conditions in dataloader:
            inputs, targets, conditions = inputs.to(device), targets.to(device), conditions.to(device)
            t_now = time.time()
            phase = clock.phase_at(t_now)

            logits = model(inputs, conditions, t_now)
            loss = criterion(logits.view(-1, cfg_shared['vocab_size']), targets.view(-1), t=t_now)
            
            if clock.weight_for_phase(phase) > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{cfg_training['epochs']} | Loss: {avg_loss:.4f} | Phase: {phase.name}")

    print("🌱 Mycelial Spiralformer training complete.")

    # --- Save the final model ---
    final_model_path = model_dir / cfg_model['save_paths']['latest_model']
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved to: {final_model_path}")


if __name__ == "__main__":
    # Allow choosing the configuration from the command line
    import argparse
    parser = argparse.ArgumentParser(description="Train the Mycelial Spiralformer.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="femto_mycelial_cpu", 
        help="The configuration to use from spiralformer_parameters.yml"
    )
    args = parser.parse_args()
    main(config_name=args.config) 