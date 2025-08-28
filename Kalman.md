# Kalman

Kalman is mainly used for tracking a system state from (sparse) measurements.

See a [face tracking](kalman_face_tracking.ipynb) example. 


## Study material

Pateraki:

[EKF tutorial](https://simondlevy.github.io/ekf-tutorial/)

[Understanding Kalman](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/kalman/kleeman_understanding_kalman.pdf)

Η διαφάνεια 13 έχει ένα παράδειγμα ενός αντικειμένου που πέφτει. -χρήσιμο

[Python tutorial](https://medium.com/@jaems33/understanding-kalman-filters-with-python-2310e87b8f48)

[Cyrill Stachniss Kalman tutorial](https://www.youtube.com/watch?v=E-6paM_Iwfc)

[RNN for tracking](https://ojs.aaai.org/index.php/AAAI/article/view/11194) Milan, A., Rezatofighi, S. H., Dick, A., Reid, I., & Schindler, K. (2017, February). Online multi-target tracking using recurrent neural networks. In Proceedings of the AAAI conference on Artificial Intelligence (Vol. 31, No. 1).

[another RNN for tracking](https://web.stanford.edu/class/cs231a/prev_projects_2016/final_report%20(7).pdf) Fang, K. (2016). Track-RNN: Joint detection and tracking using recurrent neural networks. In Proceedings of the 29th Conference on Neural Information Processing Systems (NIPS 2016), Barcelona.

[RNN training with roll-outs](https://arxiv.org/pdf/1506.03099) Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28.

1D example

    #An object is moving in one dimension, and its position is measured every second. 
    # The measurements are noisy.

    import numpy as np
    import matplotlib.pyplot as plt

    # Time step
    dt = 1.0

    # Define matrices
    A = np.array([[1, dt],  # State transition matrix
                [0, 1]])  # Updates position and velocity

    B = np.array([[0.5 * dt**2],  # Control matrix (assumes acceleration input)
                [dt]])

    C = np.array([[1, 0]])  # Measurement matrix (measuring position only)

    Q = np.array([[0.1, 0],  # Process noise covariance matrix
                [0, 0.1]])

    R = np.array([[4]])  # Measurement noise covariance (variance of noise)

    # Initialize state and covariance
    x = np.array([[0],   # Initial position
                [1]])  # Initial velocity

    P = np.eye(2)  # Initial uncertainty (identity matrix)

    # Generate simulated true positions, velocities, and noisy measurements
    np.random.seed(42)  # For reproducibility
    true_positions = []
    true_velocities = []
    measurements = []

    true_position = 0.0
    true_velocity = 1.0
    acceleration = 0.0  # No control input for simplicity

    for _ in range(20):
        # Update true position and velocity
        true_position += true_velocity * dt + 0.5 * acceleration * dt**2
        true_velocity += acceleration * dt
        true_positions.append(true_position)
        true_velocities.append(true_velocity)
        
        # Simulate noisy measurement
        measurement = true_position + np.random.normal(0, np.sqrt(R[0, 0]))
        measurements.append(measurement)

    # Run Kalman Filter
    estimates = []

    for measurement in measurements:
        # Prediction step
        u = np.array([[acceleration]])  # Control input
        x = A @ x + B @ u  # Predict next state
        P = A @ P @ A.T + Q  # Update state uncertainty with process noise

        # Measurement update step
        z = np.array([[measurement]])  # Current measurement
        y = z - C @ x  # Measurement residual
        S = C @ P @ C.T + R  # Residual covariance
        K = P @ C.T @ np.linalg.inv(S)  # Kalman Gain

        x = x + K @ y  # Update state estimate
        P = (np.eye(2) - K @ C) @ P  # Update covariance estimate

        # Store the position estimate
        estimates.append(x[0, 0])
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(true_positions, label="True Position", marker='o')
    plt.plot(measurements, label="Measurements", linestyle='--', marker='x')
    plt.plot(estimates, label="Kalman Filter Estimate", marker='s')
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.title("1D Kalman Filter with Full State-Space Representation")
    plt.grid()
    plt.show()
