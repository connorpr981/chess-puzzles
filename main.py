import mlflow
import dspy
from dspy.evaluate import Evaluate
from src.loader import load_devset
import yaml

# 1. Load config and data
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

devset = load_devset()

# 2. Define signature
class ChessMove(dspy.Signature):
    """
    The best move for a given chess puzzle position.
    Input: Chess position in FEN notation and possible legal moves
    Output: Index of the best move from the provided options
    """
    puzzle: str = dspy.InputField(description="The current position FEN")
    possible_moves: dict[str, str] = dspy.InputField(description="Possible moves {index: uci_move}")
    move: int = dspy.OutputField(description="Index of the best move in possible_moves")

# 3. Define module
class ChessMoveSelection(dspy.Module):
    """
    DSPy module that predicts the best chess move using an LLM.
    Uses the ChessMove signature to structure the input/output format.
    """
    def __init__(self):
        # Initialize the prediction component with the ChessMove signature
        self.chess_move = dspy.ChainOfThought(ChessMove)

    def forward(self, puzzle: str, possible_moves: dict[str, str]):
        # Process a single chess position and return the predicted best move
        prediction = self.chess_move(puzzle=puzzle, possible_moves=possible_moves)
        return prediction

# 4. Define validation metric
def validate_move(example, pred, trace=None):
    """
    Validation function that checks if the predicted move matches the expected move.
    Returns True if the prediction is correct, False otherwise.
    """
    return int(pred.move) == int(example["expected_move"])

# 5. Evaluate with MLflow experiment
experiment_name = "llm_chess_cot"
mlflow.set_experiment(experiment_name)

# Loop through each model in config
for model_config in config['language_models']:
    model_name = model_config['name']
    lm_config = model_config['config']
    
    with mlflow.start_run(run_name=model_name):
        # Configure the language model for DSPy
        lm = dspy.LM(**lm_config)
        dspy.configure(lm=lm)
        mlflow.dspy.autolog()

        # Log model configuration
        mlflow.log_params(lm_config)

        # Set up the evaluator with specific configuration
        evaluator = Evaluate(
            devset=devset,  # Development dataset for evaluation
            metric=validate_move,  # Validation function
            num_threads=10,  # Parallel processing threads
            display_progress=True,  # Show progress bar
            display_table=5,  # Display first 5 examples
            return_all_scores=True,  # Get individual scores
            return_outputs=True,  # Get detailed outputs
        )

        # Run evaluation and collect results
        aggregated_score, outputs, all_scores = evaluator(ChessMoveSelection())

        # Extract predictions and ground truth for logging, handling failed predictions
        predicted_moves = []
        puzzle_fens = []
        expected_moves = []
        correct_flags = []

        for example, pred, trace in outputs:
            # Skip failed predictions
            if not hasattr(pred, 'move'):
                continue
                
            predicted_moves.append(pred.move)
            puzzle_fens.append(example["puzzle"])
            expected_moves.append(example["expected_move"])
            
        # Filter all_scores to match the successful predictions
        correct_flags = [flag for i, flag in enumerate(all_scores) if i < len(predicted_moves)]

        # Log results to MLflow
        mlflow.log_metric("validation_score", aggregated_score)
        
        # Only log results for successful predictions
        if predicted_moves:  # Check if we have any valid predictions
            mlflow.log_table(
                {
                    "PuzzleFEN": puzzle_fens,
                    "ExpectedMove": expected_moves,
                    "PredictedMove": predicted_moves,
                    "Correct": correct_flags,
                },
                artifact_file=f"eval_results_{model_name}.json",
            )

        print(f"Completed evaluation for {model_name}. Validation score: {aggregated_score}")
        print(f"Successfully processed {len(predicted_moves)} out of {len(outputs)} predictions")

print("All evaluations completed. Check MLflow UI for logs and artifacts.")