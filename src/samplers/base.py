import pandas as pd
import numpy as np
import dspy
import chess
from typing import List, Dict, Optional

class BaseSampler:
    """Base class for all puzzle samplers with shared functionality."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the base sampler.
        
        Parameters:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def _create_dspy_example(self, puzzle: pd.Series, control_group: Optional[str] = None) -> Optional[List[dspy.Example]]:
        """
        Convert a puzzle DataFrame row into a sequence of DSPy Examples, one for each position.
        
        Parameters:
            puzzle: DataFrame row containing puzzle data
            control_group: Optional control group label
            
        Returns:
            List of DSPy Examples or None if conversion fails
        """
        # Get all moves from the puzzle
        moves = puzzle['Moves'].split()
        if not moves:
            return None
            
        # Create metadata dictionary
        metadata = {
            'rating': puzzle['Rating'],
            'popularity': puzzle['Popularity'],
            'rating_deviation': puzzle['RatingDeviation'],
            'themes': puzzle['Themes'],
            'moves': moves,  # Store full sequence of moves
            'initial_fen': puzzle['FEN']  # Store initial position
        }
        
        if control_group:
            metadata['control_group'] = control_group
            
        # Generate examples for each position in the sequence
        examples = []
        board = chess.Board(puzzle['FEN'])
        
        for move_idx, expected_move in enumerate(moves):
            # Get legal moves for current position
            legal_moves = list(board.legal_moves)
            legal_moves_uci = [move.uci() for move in legal_moves]
            possible_moves = {i: move_str for i, move_str in enumerate(legal_moves_uci)}
            
            # Create example for this position
            example = dspy.Example(
                puzzle=board.fen(),
                possible_moves=possible_moves,
                expected_move=expected_move,
                puzzle_id=puzzle['PuzzleId'],
                metadata=metadata,
                move_index=move_idx  # Track position in sequence
            ).with_inputs('puzzle', 'possible_moves')
            
            examples.append(example)
            
            # Make the move on the board for next position
            if move_idx < len(moves) - 1:  # Don't need to make move after last position
                board.push_uci(expected_move)
        
        return examples
    
    def _create_examples_from_df(self, df: pd.DataFrame, control_groups: Optional[pd.Series] = None) -> List[dspy.Example]:
        """
        Convert a DataFrame of puzzles into DSPy Examples.
        
        Parameters:
            df: DataFrame containing puzzles
            control_groups: Optional series of control group labels
            
        Returns:
            List of DSPy Examples
        """
        all_examples = []
        for idx, puzzle in df.iterrows():
            control_group = control_groups.iloc[idx] if control_groups is not None else None
            puzzle_examples = self._create_dspy_example(puzzle, control_group)
            if puzzle_examples:
                all_examples.extend(puzzle_examples)
        return all_examples
    
    def analyze_distribution(
        self,
        population_df: pd.DataFrame,
        sampled_examples: List[dspy.Example]
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze the distribution of characteristics in the sample compared to the population.
        
        Parameters:
            population_df: Original DataFrame containing all puzzles
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
                'rating_deviation': ex.metadata['rating_deviation'],
                'themes': ex.metadata['themes']
            })
        
        sample_df = pd.DataFrame(sample_data)
        
        # Calculate distributions
        distributions = {}
        
        # Rating distribution statistics
        pop_rating_stats = population_df['Rating'].describe()
        sample_rating_stats = sample_df['rating'].describe()
        distributions['rating_stats'] = pd.DataFrame({
            'population': pop_rating_stats,
            'sample': sample_rating_stats
        })
        
        # Popularity distribution statistics
        pop_popularity_stats = population_df['Popularity'].describe()
        sample_popularity_stats = sample_df['popularity'].describe()
        distributions['popularity_stats'] = pd.DataFrame({
            'population': pop_popularity_stats,
            'sample': sample_popularity_stats
        })
        
        return distributions 