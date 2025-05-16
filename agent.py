# agent_app.py

from load_models import ollama_model
#from load_models import anthropic_model
from pydantic_ai import Agent, RunContext
from pathlib import Path
from typing import Tuple, Dict, Any

# import your already-defined schemas and task wrappers:
from schemas import (
    PreprocessInput, PreprocessOutput,
    TrainInput,     TrainOutput,
    GenerateInput,  GenerateOutput,
    EvaluateInput,  EvaluateOutput,
    ReportInput,    ReportOutput,
)
from tasks.preprocess import preprocess_task as _pp
from tasks.train      import train_model     as _train
from tasks.generate   import generate_task    as _gen
from tasks.evaluate   import evaluate_task    as _eval
from tasks.report     import report_task      as _report

# 1) One Agent instance, listing *exactly* the names of your tools:

agent = Agent(
    name="ImagePipelineAgent",
    model= ollama_model,
    system_prompt=(
        "You are an AI assistant."
        "Yoou have five tools / functions at your disposal: preprocess_data, train_model, generate_samples, evaluate_samples and create_report."
        "Each of these functions possess arguments."
        "The user will give you instructions to run one of these tools with a specific argument."
    ),
    deps_type=None,
)





# 2) Tool decorators (as you already have them):
@agent.tool
def preprocess_data(
    ctx: RunContext,
    raw_data_path: str,
    normalize: bool = True,
    resize: Tuple[int, int] = (64, 64),
    batch_size: int = 4,
    split_ratio: float = 0.8,
    preprocessed_path: str = "preprocessed_data/noise_images",
) -> Dict[str, Tuple[str, str]]:
    inp = PreprocessInput(
        raw_data_path=Path(raw_data_path),
        normalize=normalize,
        resize=resize,
        batch_size=batch_size,
        split_ratio=split_ratio,
        preprocessed_path=Path(preprocessed_path),
    )
    out: PreprocessOutput = _pp(inp)
    train_p, test_p = out.split_train_test
    return {"split_train_test": (str(train_p), str(test_p))}

@agent.tool
def train_model(
    ctx: RunContext,
    train_path: str = "preprocessed_data/noise_images/train_loader.pt",
    test_path: str  = "preprocessed_data/noise_images/test_loader.pt",
    image_size: Tuple[int, int] = (400, 400),
    in_channel: int = 1,
    base_dim: int = 16,
    dim_mults: Tuple[int, int] = (2, 4),
    timesteps: int = 100,
    total_steps_factor: int = 256,
    max_epochs: int = 100_000,
    model_name: str = "model",
) -> Dict[str, str]:
    """
    Trains a diffusion model with the required arguments.

    Args:
        train_path: Path to the training loader as string.
        test_path: Path to the testing loader as string.
        image_size: The shape of the image in a tuple, i.e, (height, width)
        in_channel: Number of channels in the image.
        base_dim: Base dimension for the hidden layers.
        dim_mults: Progression of the hidden layers as multiples.
        timesteps: Number of diffusion steps.
        total_steps_factor: Factor that multiplies the number of timesteps.
        max_epochs: A max limit on the number of epochs to be run.
        model_name: Name of the model to be saved. 
    """

    split_train_test = [train_path, test_path]
    inp = TrainInput(
        split_train_test=(Path(split_train_test[0]), Path(split_train_test[1])),
        image_size=image_size,
        in_channel=in_channel,
        base_dim=base_dim,
        dim_mults=dim_mults,
        timesteps=timesteps,
        total_steps_factor=total_steps_factor,
        max_epochs=max_epochs,
        model_name=model_name,
    )
    out: TrainOutput = _train(inp)
    return {"model_checkpoint_path": str(out.model_checkpoint_path)}

@agent.tool
def generate_samples(
    ctx: RunContext,
    model_checkpoint_path: str = "trained_models/model.ckpt",      # no default → required
    n_samples: int = 16,
    samples_name: str = "sample",
) -> Dict[str, str]:
    """
    Wrapper around generate_task(input: GenerateInput) → GenerateOutput.
    """
    # 1) Validate that the path actually points to a file
    ckpt_path = Path(model_checkpoint_path)
    if not ckpt_path.is_file():
        raise ValueError(f"Model checkpoint not found: {model_checkpoint_path}")

    # 2) Build your Pydantic input
    inp = GenerateInput(
        model_checkpoint_path=ckpt_path,
        n_samples=n_samples,
        samples_name=samples_name,
    )
    # 3) Call the real function
    out: GenerateOutput = _gen(inp)
    # 4) Return JSON
    return {"gen_images_path": str(out.gen_images_path)}

@agent.tool
def evaluate_samples(
    ctx: RunContext,
    train_path: str = "preprocessed_data/noise_images/train_loader.pt",
    test_path: str  = "preprocessed_data/noise_images/test_loader.pt",
    n_test_data: int = 16,
    bootstrap_data: bool = True,
    kernel_patch_size: int = 3,
    num_permutations: int = 100,
    gen_images_path: str = "",
    evaluate_results_name: str = "Evaluation",
) -> Dict[str, str]:
    
    """
    Evaluates trained samples.

    Args:
        train_path: Path to the training loader as string.
        test_path: Path to the testing loader as string.
        n_test_data: Number of images to be tested.
        bootstrap_data: Boolean variable that decides to run bootstraping on data.
        kernel_patch_size: Kernel size of the convolutional layer to calculate distance.
        num_permutations: Number of permutations to be run.
        gen_images_path: Location where images should are saved.
        evaluate_results_name: Name of the evaluation results. 
    """
    
    split_train_test = [train_path, test_path]
    split_train_test=(Path(split_train_test[0]), Path(split_train_test[1]))
    inp = EvaluateInput(
        split_train_test=(Path(split_train_test[0]), Path(split_train_test[1])),
        n_test_data=n_test_data,
        bootstrap_data=bootstrap_data,
        kernel_patch_size=kernel_patch_size,
        num_permutations=num_permutations,
        gen_images_path=Path(gen_images_path),
        evaluate_results_name=evaluate_results_name,
    )
    out: EvaluateOutput = _eval(inp)
    return {"stats_csv": str(out.stats_csv), "plot_png": str(out.plot_png)}

@agent.tool
def create_report(
    ctx: RunContext,
    gen_images_path: str,
    stats_csv: str,
    plot_png: str,
    text_data: Dict[str, Any] = {
            "introduction": "This is an automated report that displays various artifacts including images, a table, and a plot.",
            "image_description": "The images displayed above are sampled from the provided .pth file.",
            "table_description": "The table below is extracted from the provided .csv file.",
            "plot_description": "This plot is the .png image provided for visualization."
        },
    report_name: str = "report",
) -> Dict[str, str]:
    inp = ReportInput(
        gen_images_path=Path(gen_images_path),
        stats_csv=Path(stats_csv),
        plot_png=Path(plot_png),
        text_data=text_data,
        report_name=report_name,
    )
    out: ReportOutput = _report(inp)
    return {"report_path": str(out.report_path)}

# 3) A simple REPL to test
if __name__ == "__main__":

  

    result = agent.run_sync(
        "Hello, please run the function preprocess_data with variables raw_data_path = 'Ellipses_1', preprocessed_path = 'preprocessed_data/one_batch_size', batch_size = 1 and split_ratio = 0.8. ",
        #"Hello, please run the function train_model with variables train_path = 'preprocessed_data/one_batch_size/train_loader.pt', test_path =  'preprocessed_data/one_batch_size/test_loader.pt', max_epochs = 100 and model_name = model_v3. For the other variables, use the default arguments defined in the function.",
        #"Hello, please run the function generate_samples with variables n_samples = 4, samples_name = 'new_samples' and model_checkpoint_path: str = 'trained_models/model_v3.ckpt'. ",
        #"Hello, please run the function evaluate_samples with variables train_path = 'preprocessed_data/ellipses/train_loader.pt', test_path = 'preprocessed_data/ellipses/test_loader.pt', evaluate_results_name = 'one_batch_size' and gen_images_path = 'generated_samples/sample_v2.pt, n_test_data = 16 and num_permutations = 1000. ",
        #"Hello, please run the function create_report with variables gen_images_path = 'generated_samples/new_samples.pt', stats_csv = 'Evaluation Results/one_batch_size/perm_stats.csv', plot_png = 'Evaluation Results/one_batch_size/perm_plot.png'.  ",
        
        
        
        
        deps = None
    )

    print("---------------")
    print("Result:")
    print(result.output)