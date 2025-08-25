import mlflow
from mlflow.tracking import MlflowClient


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def get_last_run_with_prefix(experiment_name: str, run_prefix: str):
    """
    Retrieve the most recent MLflow run within a specified experiment that has a name starting with a given prefix.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment to search within.
    - run_prefix (str): Prefix string that the run name should start with.

    Returns:
    - mlflow.entities.Run or None: The most recent MLflow run with the specified prefix, or None if no such run exists.

    Raises:
    - ValueError: If the specified experiment does not exist.
    """
    client = MlflowClient()

    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"get_last_run_with_prefix(): Experiment '{experiment_name}' not found!"
        )

    # Search runs in this experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName LIKE '{run_prefix}%'",
        order_by=["start_time DESC"],  # newest first
        max_results=1,
    )

    if not runs:
        return None
    return runs[0]
