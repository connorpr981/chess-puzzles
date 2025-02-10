import datasets
import os
import pandas as pd
import dspy
from typing import List

def load_devset() -> List[dspy.Example]:
    """Load existing devset."""
    devset_path = "data/devset.parquet"
    
    if not os.path.exists(devset_path):
        raise FileNotFoundError("Devset file not found at data/devset.parquet")
        
    print("Loading existing devset...")
    devset_df = pd.read_parquet(devset_path)
    examples = [
        dspy.Example(
            puzzle=row['puzzle'],  # This is now the FEN after opponent's move
            possible_moves={str(k): v for k, v in eval(row['possible_moves']).items()},  # Ensure string keys
            expected_move=row['expected_move'],
            puzzle_id=row['puzzle_id'],
            metadata={
                'rating': row['rating'],
                'popularity': row['popularity'],
                'rating_deviation': row['rating_deviation'],
                'themes': row['themes'],
                'initial_fen': row['initial_fen'],  # Original FEN before opponent's move
                'opponent_move': row['opponent_move'],  # Store opponent's move for reference
            }
        ).with_inputs('puzzle', 'possible_moves')
        for _, row in devset_df.iterrows()
    ]
    
    if not examples:
        raise ValueError("No examples found in existing devset")
    
    print(f"Loaded {len(examples)} examples from existing devset")
    return examples 