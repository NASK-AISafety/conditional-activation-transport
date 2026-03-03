from typing import Optional
import logging

import torch

from .methods import CONDITIONING_METHODS, SteeringMethod

logger = logging.getLogger(__name__)

ACTIVATION_FUNCTIONS = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "gelu": torch.nn.GELU,
}


class SteeringTrainingDataset(torch.utils.data.Dataset):
    """Dataset for training steering transport models.

    Args:
        from_activations: Source activations tensor of shape (N, D) where N is batch size, D is hidden dimension
        to_activations: Target activations tensor of shape (N, D)

    Raises:
        ValueError: If activation shapes don't match or are invalid
    """

    def __init__(self, from_activations: torch.Tensor, to_activations: torch.Tensor):
        if from_activations.dim() != 2 or to_activations.dim() != 2:
            raise ValueError(
                f"Expected 2D tensors, got from_activations: {from_activations.shape}, "
                f"to_activations: {to_activations.shape}"
            )
        if from_activations.shape != to_activations.shape:
            raise ValueError(
                f"Activation shapes must match. Got from: {from_activations.shape}, "
                f"to: {to_activations.shape}"
            )

        self.from_activations = from_activations.to(torch.float32)
        self.to_activations = to_activations.to(torch.float32)
        logger.info(
            "Created dataset with %d samples, hidden_size=%d",
            len(self),
            from_activations.size(1),
        )

    def __len__(self) -> int:
        return self.from_activations.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.from_activations[idx], self.to_activations[idx]


class MLPTransport(torch.nn.Module):
    """Multi-layer perceptron transport model for activation steering.

    Args:
        input_size: Dimension of input/output activations
        hidden_size: Dimension of hidden layers
        activation: Activation function name ('relu', 'tanh', 'sigmoid', 'gelu')
        use_residual: Whether to add residual connection
        use_layer_norm: Whether to use layer normalization
        num_hidden_layers: Number of hidden layers (default: 2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        activation: str = "gelu",
        use_residual: bool = True,
        use_layer_norm: bool = True,
        num_hidden_layers: int = 2,
        zero_init_output: bool = True,
    ):
        super(MLPTransport, self).__init__()
        if activation not in ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unknown activation: {activation}. Choose from {list(ACTIVATION_FUNCTIONS.keys())}"
            )
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

        self.use_residual = use_residual
        layers = []

        # Input layer: input_size -> hidden_size
        layers.extend(
            [
                torch.nn.Linear(input_size, hidden_size),
                ACTIVATION_FUNCTIONS[activation](),
            ]
        )
        if use_layer_norm:
            layers.append(torch.nn.RMSNorm(hidden_size))

        # Hidden layers: hidden_size -> hidden_size (repeated num_hidden_layers - 1 times)
        for _ in range(num_hidden_layers - 1):
            layers.extend(
                [
                    torch.nn.Linear(hidden_size, hidden_size),
                    ACTIVATION_FUNCTIONS[activation](),
                ]
            )
            if use_layer_norm:
                layers.append(torch.nn.RMSNorm(hidden_size))

        # Output layer: hidden_size -> input_size
        output_layer = torch.nn.Linear(hidden_size, input_size)
        if zero_init_output:
            torch.nn.init.zeros_(output_layer.weight)
            torch.nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP transport.

        Args:
            x: Input tensor of shape (..., input_size)

        Returns:
            Transported tensor of same shape as input
        """
        out = self.model(x)
        if self.use_residual:
            return x + out
        return out


class AffineTransport(torch.nn.Module):
    """Affine transport model: T(x) = Ax + b"""

    def __init__(self, hidden_size: int):
        super(AffineTransport, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        # Initialize to identity
        torch.nn.init.eye_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_transport(
    transport_model: torch.nn.Module,
    dataset: SteeringTrainingDataset,
    num_epochs: int = 1000,
    batch_size: int = 128,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
    gradient_clip: float = 1.0,
    patience: int = 50,
    min_delta: float = 1e-5,
    verbose: bool = True,
    regularization_weight: float = 0.0,
) -> tuple[torch.nn.Module, dict]:
    """Train a transport model using MSE loss.

    Args:
        transport_model: The model to train
        dataset: Training dataset
        num_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization weight
        device: Device to train on (defaults to cuda if available)
        gradient_clip: Maximum gradient norm (0 to disable)
        patience: Early stopping patience (0 to disable)
        min_delta: Minimum loss improvement for early stopping
        verbose: Whether to print training progress
        regularization_weight: Weight for T(x)=x regularization on target distribution

    Returns:
        Tuple of (trained_model, training_metrics_dict)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) > batch_size,
    )
    transport_model.to(device)
    optimizer = torch.optim.Adam(
        transport_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
    )
    loss_fn = torch.nn.MSELoss()

    # Training metrics
    metrics = {
        "train_losses": [],
        "best_loss": float("inf"),
        "best_epoch": 0,
    }

    # Early stopping
    epochs_without_improvement = 0
    best_loss = float("inf")

    logger.info(
        "Starting training on %s with %d samples, batch_size=%d, lr=%f, reg_weight=%f",
        device,
        len(dataset),
        batch_size,
        learning_rate,
        regularization_weight,
    )

    transport_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for from_activations, to_activations in dataloader:
            from_activations = from_activations.to(device)
            to_activations = to_activations.to(device)

            optimizer.zero_grad()
            transformed_activations = transport_model(from_activations)
            loss = loss_fn(transformed_activations, to_activations)

            if regularization_weight > 0:
                # Regularization: T(to_activations) should be close to to_activations
                reg_output = transport_model(to_activations)
                reg_loss = loss_fn(reg_output, to_activations)
                loss = loss + regularization_weight * reg_loss

            loss.backward()

            # Gradient clipping for stability
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    transport_model.parameters(), gradient_clip
                )

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        metrics["train_losses"].append(avg_loss)
        scheduler.step(avg_loss)

        # Check for improvement
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            metrics["best_loss"] = best_loss
            metrics["best_epoch"] = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            logger.info(
                "Epoch %d/%d: loss=%.6f, best_loss=%.6f (epoch %d)",
                epoch + 1,
                num_epochs,
                avg_loss,
                best_loss,
                metrics["best_epoch"] + 1,
            )

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    transport_model.eval()
    logger.info("Training completed. Best loss: %.6f", metrics["best_loss"])
    return transport_model, metrics


class MLPTransportSteering(SteeringMethod):
    """Steering method using MLP transport model.

    Learns a non-linear MLP transformation between source and target activation distributions.

    Args:
        steering_mode: Mode for steering application
        conditioning_type: Type of conditioning ('none', 'MAHALANOBIS', etc.)
        hidden_size: Size of MLP hidden layers (default: 128)
        use_residual: Whether to use residual connections (default: False)
        use_layer_norm: Whether to use layer normalization (default: True)
        num_hidden_layers: Number of hidden layers in the MLP (default: 2)
        **conditioner_kwargs: Additional arguments for conditioning method
    """

    def __init__(
        self,
        steering_mode,
        conditioning_type: str = "none",
        hidden_size: int = 128,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        num_hidden_layers: int = 2,
        zero_init_output: bool = True,
        **conditioner_kwargs,
    ):
        self.steering_mode = steering_mode
        self.hidden_size = hidden_size
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.num_hidden_layers = num_hidden_layers
        self.zero_init_output = zero_init_output
        self.transport_model: Optional[MLPTransport] = None
        self.conditioning_type = conditioning_type
        self.training_metrics: Optional[dict] = None
        if conditioning_type == "none":
            self.conditioner = None
        else:
            if conditioning_type not in CONDITIONING_METHODS:
                raise ValueError(
                    f"Unknown conditioning type: {conditioning_type}. "
                    f"Choose from {list(CONDITIONING_METHODS.keys())}"
                )
            self.conditioner = CONDITIONING_METHODS[conditioning_type](
                **conditioner_kwargs
            )

    def train(
        self,
        from_activations: torch.Tensor,
        to_activations: torch.Tensor = None,
        regularization_weight: float = 0.0,
        **train_kwargs,
    ) -> None:
        """Train the MLP transport model.

        Args:
            from_activations: Source activations of shape (N, D)
            to_activations: Target activations of shape (N, D)
            regularization_weight: Weight for T(x)=x regularization on target distribution
            **train_kwargs: Additional arguments passed to train_transport
        """
        if to_activations is None:
            raise ValueError("to_activations must be provided for transport training")

        if from_activations.dim() != 2 or to_activations.dim() != 2:
            raise ValueError(
                f"Expected 2D activation tensors, got shapes: "
                f"from={from_activations.shape}, to={to_activations.shape}"
            )

        input_size = from_activations.size(1)
        transport_model = MLPTransport(
            input_size=input_size,
            hidden_size=self.hidden_size,
            use_residual=self.use_residual,
            use_layer_norm=self.use_layer_norm,
            num_hidden_layers=self.num_hidden_layers,
            zero_init_output=self.zero_init_output,
        )

        dataset = SteeringTrainingDataset(from_activations, to_activations)
        self.transport_model, self.training_metrics = train_transport(
            transport_model,
            dataset,
            regularization_weight=regularization_weight,
            **train_kwargs,
        )

        # Train conditioner if present
        if self.conditioner is not None:
            self.conditioner.train(from_activations, to_activations)

        logger.info(
            "MLP transport trained with best loss: %.6f",
            self.training_metrics["best_loss"],
        )

    def steer(self, x: torch.Tensor, strength: float = 1.0, **_kwargs) -> torch.Tensor:
        """Apply MLP transport steering.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            strength: Steering strength (0 to 1)
            **_kwargs: Additional arguments (unused)

        Returns:
            Steered activations of same shape as input
        """
        if self.transport_model is None:
            raise RuntimeError(
                "Model must be trained before steering. Call train() first."
            )

        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (B, L, D), got shape {x.shape}")

        # Move model to correct device
        self.transport_model.to(x.device)

        if self.conditioner is not None:
            # conditioner.condition expects mean over sequence dim
            mask = self.conditioner.condition(x)
        else:
            mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        if not mask.any():
            logger.warning("No samples passed conditioning, skipping steering")
            return x

        # Shape: (batch_size, hidden_size)
        current_context_mean = x[mask].mean(dim=1).float()

        with torch.no_grad():
            transported_mean = self.transport_model(current_context_mean)

        steering_vector = (transported_mean - current_context_mean).to(x.dtype)

        # Expand steering_vector to match sequence dimension
        # Shape: (num_masked, 1, hidden_size) -> broadcasts to (num_masked, seq_len, hidden_size)
        x[mask] = x[mask] + (strength * steering_vector.unsqueeze(1))

        return x


class AffineTransportSteering(SteeringMethod):
    """Steering method using Affine transport model."""

    def __init__(
        self,
        steering_mode,
        conditioning_type: str = "none",
        **conditioner_kwargs,
    ):
        self.steering_mode = steering_mode
        self.transport_model: Optional[AffineTransport] = None
        self.conditioning_type = conditioning_type
        self.training_metrics: Optional[dict] = None

        if conditioning_type == "none":
            self.conditioner = None
        else:
            if conditioning_type not in CONDITIONING_METHODS:
                raise ValueError(
                    f"Unknown conditioning type: {conditioning_type}. "
                    f"Choose from {list(CONDITIONING_METHODS.keys())}"
                )
            self.conditioner = CONDITIONING_METHODS[conditioning_type](
                **conditioner_kwargs
            )

    def train(
        self,
        from_activations: torch.Tensor,
        to_activations: torch.Tensor = None,
        regularization_weight: float = 0.0,
        **train_kwargs,
    ) -> None:
        if to_activations is None:
            raise ValueError("to_activations must be provided for transport training")

        if from_activations.dim() != 2 or to_activations.dim() != 2:
            raise ValueError(
                f"Expected 2D activation tensors, got shapes: "
                f"from={from_activations.shape}, to={to_activations.shape}"
            )

        input_size = from_activations.size(1)
        transport_model = AffineTransport(hidden_size=input_size)

        dataset = SteeringTrainingDataset(from_activations, to_activations)
        self.transport_model, self.training_metrics = train_transport(
            transport_model,
            dataset,
            regularization_weight=regularization_weight,
            **train_kwargs,
        )

        if self.conditioner is not None:
            self.conditioner.train(from_activations, to_activations)

        logger.info(
            "Affine transport trained with best loss: %.6f",
            self.training_metrics["best_loss"],
        )

    def steer(self, x: torch.Tensor, strength: float = 1.0, **_kwargs) -> torch.Tensor:
        if self.transport_model is None:
            raise RuntimeError(
                "Model must be trained before steering. Call train() first."
            )

        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (B, L, D), got shape {x.shape}")

        self.transport_model.to(x.device)

        if self.conditioner is not None:
            mask = self.conditioner.condition(x)
        else:
            mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        if not mask.any():
            return x

        current_context_mean = x[mask].mean(dim=1).float()
        with torch.no_grad():
            transported_mean = self.transport_model(current_context_mean)

        steering_vector = (transported_mean - current_context_mean).to(x.dtype)
        x[mask] = x[mask] + (strength * steering_vector.unsqueeze(1))
        return x
