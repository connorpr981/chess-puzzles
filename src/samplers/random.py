import pandas as pd
import dspy
import chess
from typing import List, Dict
from .base import BaseSampler

class RandomSampler(BaseSampler):
    """Simple random sampler for single-move chess puzzles."""
    
    def create_sample(
        self,
        df: pd.DataFrame,
        sample_size: int = 1000
    ) -> List[dspy.Example]:
        """
        Create a simple random sample from single-move chess puzzles.
        
        Parameters:
            df: DataFrame containing the puzzles (already filtered for single moves)
            sample_size: Number of puzzles to sample
            
        Returns:
            List of DSPy Examples containing the sampled puzzles
        """
        # Take random sample
        sampled_df = df.sample(n=min(sample_size, len(df)), random_state=self.random_state)
        
        # Convert to DSPy Examples
        examples = []
        for _, row in sampled_df.iterrows():
            try:
                # Create board from FEN after opponent's move
                board = chess.Board(row['puzzle_fen'])
                
                # Get all legal moves in UCI format
                legal_moves = {str(i): move.uci() for i, move in enumerate(board.legal_moves)}
                
                if not legal_moves:
                    continue  # Skip if no legal moves
                    
                example = dspy.Example(
                    puzzle=row['puzzle_fen'],  # FEN after opponent's move
                    possible_moves=legal_moves,
                    expected_move=row['expected_move'],
                    puzzle_id=row['PuzzleId'],
                    metadata={
                        'rating': row['Rating'],
                        'popularity': row['Popularity'],
                        'rating_deviation': row['RatingDeviation'],
                        'themes': row['Themes'],
                        'initial_fen': row['FEN'],  # Original FEN
                        'opponent_move': row['opponent_move']
                    }
                ).with_inputs('puzzle', 'possible_moves')
                
                examples.append(example)
            except Exception as e:
                print(f"Error creating example for puzzle {row['PuzzleId']}: {e}")
                continue
                
        return examples

def analyze_sample_distribution(
    df: pd.DataFrame,
    sampled_examples: List[dspy.Example]
) -> Dict[str, pd.DataFrame]:
    """
    Analyze the distribution of characteristics in the sample compared to the population.
    
    Parameters:
        df: Original DataFrame containing all puzzles
        sampled_examples: List of sampled DSPy Examples
        
    Returns:
        Dictionary containing DataFrames comparing distributions
    """
    # Convert sampled examples back to DataFrame for analysis
    sample_data = []
    for ex in sampled_examples:
        sample_data.append({
            'rating': ex.metadata['rating'],
            'popularity': ex.metadata['popularity'],
            'rating_deviation': ex.metadata['rating_deviation']
        })
    
    sample_df = pd.DataFrame(sample_data)
    
    # Calculate distributions
    distributions = {}
    
    # Rating distribution statistics
    pop_rating_stats = df['Rating'].describe()
    sample_rating_stats = sample_df['rating'].describe()
    distributions['rating_stats'] = pd.DataFrame({
        'population': pop_rating_stats,
        'sample': sample_rating_stats
    })
    
    # Popularity distribution statistics
    pop_popularity_stats = df['Popularity'].describe()
    sample_popularity_stats = sample_df['popularity'].describe()
    distributions['popularity_stats'] = pd.DataFrame({
        'population': pop_popularity_stats,
        'sample': sample_popularity_stats
    })
    
    return distributions
