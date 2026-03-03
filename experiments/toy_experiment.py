import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
import logging
import sys
import os

# Ensure we can import from utils if running from root
sys.path.append(os.getcwd())

from steering.methods import MeanSteering, SteeringMode, LinearTransportSteering
from steering.training import MLPTransportSteering

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_scenario_a(n_samples=1000):
    """
    Scenario A: Simple Gaussian (Baseline)
    Relation: Shift (Translation).
    Safe = Unsafe + Shift
    """
    # Safe: round blob at (5, 5)
    # Let's start with a base distribution and apply functions
    base = np.random.normal(loc=0, scale=0.5, size=(n_samples, 2))

    # Safe is centered at (3, 3)
    safe_mean = np.array([3.0, 3.0])
    safe = base + safe_mean

    # Unsafe is centered at (-3, -3)
    # Relation: Unsafe = Safe - 6.0
    # Or: Unsafe = base + (-3, -3)
    unsafe_mean = np.array([-3.0, -3.0])
    unsafe = base + unsafe_mean

    # The "Function" from Unsafe to Safe is f(x) = x + (6, 6)
    return torch.tensor(unsafe, dtype=torch.float32), torch.tensor(
        safe, dtype=torch.float32
    )


def generate_scenario_b(n_samples=1000):
    """
    Scenario B: Variance Mismatch (Rotation)
    Relation: Rotation + Shift (to ensure non-zero mean).
    Unsafe is rotated version of Safe (or vice versa).
    """
    # Base: Tall and narrow (Safe shape)
    # Scale x by 0.5, y by 4.0
    base = np.random.normal(size=(n_samples, 2))
    base[:, 0] *= 0.5
    base[:, 1] *= 4.0

    # Add a shift so mean is not zero
    shift = np.array([5.0, 5.0])
    safe = base + shift  # Safe is the tall oval, centered at (5,5)

    # Unsafe: Rotate Safe around the center (5,5), but then maybe shift it elsewhere or keep it there?
    # User said: "ensure that means are always different than 0 especially in variance mismatch"
    # The original "Variance Mismatch" scenario often has "shared exact same center point (mean)".
    # If they share the same mean, and that mean is non-zero, ActAdd still fails (vector is 0).
    # If they have different means, ActAdd might work (shift).
    # User text: "ActAdd: Fails completely. Since the centers are the same, Mean_Safe - Mean_Unsafe = 0. The vector is zero. Nothing moves."
    # So we MUST keep the means basically the same (or very close) to show ActAdd failure, OR we assume the "Variance Mismatch" includes a shift?
    # Actually, the user's prompt says: "ensure that means are always different than 0".
    # It DOES NOT say "means must be different from each other". It says "different than 0" (the origin).
    # So both can be centered at (5,5). Mean_Safe - Mean_Unsafe approx 0.

    # Let's create Unsafe by rotating the base (centered at 0), then adding the SAME shift.
    # Rotate 90 deg
    base_rotated = np.zeros_like(base)
    base_rotated[:, 0] = -base[:, 1]
    base_rotated[:, 1] = base[:, 0]

    unsafe = base_rotated + shift

    return torch.tensor(unsafe, dtype=torch.float32), torch.tensor(
        safe, dtype=torch.float32
    )


def generate_scenario_c(n_samples=1000):
    """
    Scenario C: The Moon (Novelty)
    Relation: Non-linear wrapping (Polar transform).
    We generate Safe (Blob) and map it to Unsafe (Moon) via a function.
    Target function (Unsafe -> Safe) is the inverse.
    """
    # Safe: Compact Gaussian blob at (1.0, -0.25)
    # Let's center `safe` base at (0,0) for easier math, then shift.
    base_s = np.random.normal(loc=0.0, scale=0.2, size=(n_samples, 2))

    # Map to Moon (Unsafe)
    # We want a crescent.
    # Let x_s, y_s be the source coordinates.
    # Map them to r, theta.
    # r = R0 + y_s
    # theta = theta0 + x_s

    x_s = base_s[:, 0]
    y_s = base_s[:, 1]

    # Parameters for the moon shape
    radius_base = 2.0
    angle_center = np.pi / 2  # Top arch
    angle_spread = 2.5  # How wide the moon is

    r = radius_base + y_s
    theta = angle_center + x_s * angle_spread

    u_x = r * np.cos(theta)
    u_y = r * np.sin(theta)

    unsafe = np.stack([u_x, u_y], axis=1)

    # Center the unsafe roughly at -0.5, 0.5 for visualization balance?
    # No, let's keep it fixed relative to Safe.
    # But we want the "Safe" to appear distinct.
    # Let "Safe" be the base_s shifted to some target location

    # The problem: If we define Safe = base_s, it is at (0,0).
    # Unsafe is at y=2.0 (top).
    # Let's shift Safe to (1.0, -0.5) to match the previous visual preference.
    target_offset = np.array([1.0, -0.5])
    safe = base_s + target_offset

    # Note: To maintain the paired function, we should calculate Unsafe FROM the final Safe.
    # Unsafe = f(Safe - offset).
    # Here we generated Unsafe from base_s, and Safe from base_s. They are paired sample-wise.

    return torch.tensor(unsafe, dtype=torch.float32), torch.tensor(
        safe, dtype=torch.float32
    )


def generate_scenario_d(n_samples=1000):
    """
    Scenario D: Multi-Modal XOR
    Relation: Piecewise Affine (Context Dependent Shifts).
    """
    samples_per_cluster = n_samples // 4

    clusters = [
        # (Unsafe Center, Shift Vector to Safe)
        ([-5, 5], [3, -3]),  # TL (Inward) -> Target (-2, 2)
        ([5, -5], [-3, 3]),  # BR (Inward) -> Target (2, -2)
        ([5, 5], [3, 3]),  # TR (Outward) -> Target (8, 8)
        ([-5, -5], [-3, -3]),  # BL (Outward) -> Target (-8, -8)
    ]

    unsafe_list = []
    safe_list = []

    for u_cen, shift in clusters:
        # Generate Unsafe blob
        u = np.random.normal(loc=u_cen, scale=0.5, size=(samples_per_cluster, 2))
        # Safe = Unsafe + Shift
        s = u + np.array(shift)

        unsafe_list.append(u)
        safe_list.append(s)

    unsafe = np.concatenate(unsafe_list, axis=0)
    safe = np.concatenate(safe_list, axis=0)

    return torch.tensor(unsafe, dtype=torch.float32), torch.tensor(
        safe, dtype=torch.float32
    )


def train_and_steer(method_name, method_class, unsafe_train, safe_train, unsafe_test):
    logger.info(f"Training {method_name}...")

    # Initialize method
    # Note: Using UNCONDITIONAL mode for simplicity
    if method_class in [MeanSteering, LinearTransportSteering]:
        method = method_class(
            steering_mode=SteeringMode.UNCONDITIONAL, conditioning_type="none"
        )
        method.train(unsafe_train, safe_train)
    else:  # MLP
        # These expect special initialization if we followed the codebase pattern exactly
        # But the classes have default inits that should work.
        # MLPTransportSteering init: steering_mode, conditioning_type, hidden_size=128...
        method = method_class(
            steering_mode=SteeringMode.UNCONDITIONAL,
            conditioning_type="none",
            hidden_size=32,
        )  # Smaller hidden size for 2D is enough

        # Train loop
        # Providing regularization_weight=0.0 as per Toy Experiment requirements (show pure steering capability)
        # Using a reasonable number of epochs for small 2D data
        method.train(
            unsafe_train,
            safe_train,
            num_epochs=200,
            batch_size=128,
            verbose=False,
            learning_rate=1e-2,
        )

    # Steer
    # Input expected: (Batch, Sequence, Hidden) -> (N, 1, 2)
    unsafe_test_input = unsafe_test.unsqueeze(1)

    # Apply steering with strength 1.0 (full transport)
    steered_full = method.steer(unsafe_test_input.clone(), strength=1.0)

    # Output: (N, 1, 2) -> (N, 2)
    return steered_full.squeeze(1).detach()


def run_experiment():
    scenarios = {
        "Scenario A": generate_scenario_a,
        "Scenario B": generate_scenario_b,
        "Scenario C": generate_scenario_c,
        "Scenario D": generate_scenario_d,
    }

    methods = [
        ("ActAdd (Mean)", MeanSteering),
        ("Linear-ACT", LinearTransportSteering),
        ("MLP Transport", MLPTransportSteering),
    ]

    all_results = []

    for scenario_name, generator in scenarios.items():
        logger.info(f"--- Running {scenario_name} ---")
        # Generate fresh data for training
        unsafe_train, safe_train = generator(n_samples=1000)

        # Convert to numpy for saving
        unsafe_np = unsafe_train.numpy()
        safe_np = safe_train.numpy()

        # Save Original Data
        for i in range(unsafe_np.shape[0]):
            all_results.append(
                {
                    "Scenario": scenario_name,
                    "Type": "Original_Unsafe",
                    "Method": "None",
                    "X": unsafe_np[i, 0],
                    "Y": unsafe_np[i, 1],
                }
            )
            all_results.append(
                {
                    "Scenario": scenario_name,
                    "Type": "Original_Safe",
                    "Method": "None",
                    "X": safe_np[i, 0],
                    "Y": safe_np[i, 1],
                }
            )

        # 2. Train and Steer each method
        for method_name, method_class in methods:
            try:
                # We use the same data for train/test for the 'toy' demo visualization
                steered = train_and_steer(
                    method_name, method_class, unsafe_train, safe_train, unsafe_train
                )
                steered_np = steered.numpy()

                # Save Steered Results
                for i in range(steered_np.shape[0]):
                    all_results.append(
                        {
                            "Scenario": scenario_name,
                            "Type": "Steered",
                            "Method": method_name,
                            "X": steered_np[i, 0],
                            "Y": steered_np[i, 1],
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to run {method_name} on {scenario_name}: {e}")

    # Create DataFrame and Save
    df = pd.DataFrame(all_results)
    output_path = "experiment_9/toy_experiment_results.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
