"""PyTorch training for the anatomical integration network."""

import numpy as np
from .dependencies import require_torch
from .losses import pairwise_softplus, prepare_pairwise


def resolve_device(requested="auto"):
    torch = require_torch()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pooled_prediction_variance(predictions, dataset_weights):
    """Compute sigma^2 = sum_d q_d mean((prediction_d - mean_d)^2)."""
    torch = require_torch()
    variance = torch.zeros((), dtype=torch.float32, device=next(iter(predictions.values())).device)
    for name, predicted in predictions.items():
        centered_prediction = predicted - predicted.mean()
        dataset_variance = torch.mean(centered_prediction ** 2)
        variance = variance + np.float32(dataset_weights[name]) * dataset_variance
    return variance


def prediction_variance_normalize(output, dataset_weights, epsilon=1e-8):
    """Divide the final map and its derived values by pooled prediction sigma."""
    torch = require_torch()
    variance = pooled_prediction_variance(output["predictions"], dataset_weights)
    scale = torch.sqrt(torch.clamp(variance, min=epsilon))
    normalized_predictions = {name: predicted / scale for name, predicted in output["predictions"].items()}
    normalized = output.copy()
    normalized["map"] = output["map"] / scale
    normalized["representations"] = output["representations"] / scale
    normalized["domain_maps"] = output["domain_maps"] / scale
    normalized["predictions"] = normalized_predictions
    normalized["prediction_scale"] = scale
    return normalized


def forward_loss(model, patients, pairs, dataset_weights, dataset_signs):
    """Run spatial layers, patient scoring, normalization, and the shared loss."""
    output = model(patients)
    normalized = prediction_variance_normalize(output, dataset_weights)
    loss = pairwise_softplus(normalized["predictions"], pairs, dataset_weights, dataset_signs)
    return normalized, loss


def snapshot(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def restore(model, parameters):
    torch = require_torch()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(parameters[name].to(parameter.device))


def result(model, output, loss, history, device):
    fitted = {}
    fitted["model"] = model
    fitted["map"] = output["map"].detach().cpu().numpy().astype(np.float32, copy=False)
    fitted["representations"] = output["representations"].detach().cpu().numpy().astype(np.float32, copy=False)
    fitted["domain_maps"] = output["domain_maps"].detach().cpu().numpy().astype(np.float32, copy=False)
    fitted["predictions"] = {name: values.detach().cpu().numpy().astype(np.float32, copy=False) for name, values in output["predictions"].items()}
    fitted["prediction_scale"] = float(output["prediction_scale"].detach().cpu())
    fitted["loss"] = float(loss.detach().cpu())
    fitted["history"] = history
    fitted["epochs"] = len(history)
    fitted["device"] = str(device)
    fitted["optimizer"] = "adam"
    fitted["batch_mode"] = "full"
    return fitted


def fit(model, patient_maps, outcomes, dataset_weights, dataset_signs, *, learning_rate=0.01, max_epochs=2000, tolerance=1e-7, patience=100, weight_decay=0.0, gradient_clip=5.0, device="auto", seed=2026, record_parameters=False):
    """Train with full-batch Adam using every fitting patient and non-tied pair each epoch.
    record_parameters: whether to keep every epoch's parameters (CPU copies, aligned with history) so the training can be replayed as maps.
    """
    torch = require_torch()
    if max_epochs < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("max_epochs, patience, and learning_rate must be positive.")
    torch.manual_seed(int(seed))
    selected_device = resolve_device(device)
    model = model.to(selected_device)
    patients = {name: torch.as_tensor(np.asarray(values, dtype=np.float32), device=selected_device) for name, values in patient_maps.items()}
    pairs = prepare_pairwise(outcomes, selected_device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The network has no trainable parameters.")
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate), weight_decay=float(weight_decay))
    best_loss = np.inf
    best_parameters = None
    stale = 0
    history = []
    trajectory = []

    for epoch in range(1, int(max_epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        _, loss = forward_loss(model, patients, pairs, dataset_weights, dataset_signs)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Neural loss became nonfinite at epoch {epoch}.")
        value = float(loss.detach().cpu())
        history.append({"epoch": epoch, "loss": value})
        if record_parameters:
            trajectory.append(snapshot(model))
        if best_loss - value > tolerance:
            best_loss = value
            best_parameters = snapshot(model)
            stale = 0
        else:
            stale += 1
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
        optimizer.step()
        if stale >= patience:
            break

    with torch.no_grad():
        final_output, final_loss = forward_loss(model, patients, pairs, dataset_weights, dataset_signs)
    final_value = float(final_loss.detach().cpu())
    if np.isfinite(final_value) and best_loss - final_value > tolerance:
        best_loss = final_value
        best_parameters = snapshot(model)
    if best_parameters is not None:
        restore(model, best_parameters)
    with torch.no_grad():
        output, loss = forward_loss(model, patients, pairs, dataset_weights, dataset_signs)
    fitted = result(model, output, loss, history, selected_device)
    fitted["parameter_trajectory"] = trajectory if record_parameters else None
    return fitted


def replay(model, trajectory, patient_maps, dataset_weights, indices):
    """The prediction-normalized map and representation weights at each recorded epoch in indices; the model is changed in place."""
    torch = require_torch()
    device = next(model.parameters()).device
    patients = {name: torch.as_tensor(np.asarray(values, dtype=np.float32), device=device) for name, values in patient_maps.items()}
    frames, mixes = [], []
    with torch.no_grad():
        for index in indices:
            restore(model, trajectory[index])
            frames.append(prediction_variance_normalize(model(patients), dataset_weights)["map"].cpu().numpy())
            mixes.append(model.representation_mix.cpu().numpy())
    return frames, mixes
