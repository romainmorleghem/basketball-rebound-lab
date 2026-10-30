
"""
BASKETBALL SHOT AND REBOUND SIMULATION
======================================

OVERVIEW
--------
This script implements a simplified 3D physical model of a basketball
shot, including ballistic motion, collisions with the rim and the
backboard, and Monte Carlo sampling of initial velocities.

For a fixed shooting position and nominal launch velocity, random
perturbations are independently applied to the three velocity
components. Each simulated shot is followed until it crosses the
hoop plane or reaches the ground.

The simulation records collision events, shot outcomes and landing
positions. The results are then used to study the spatial, radial
and angular distributions of missed-shot rebounds.


1. COORDINATE SYSTEM AND PHYSICAL MODEL
---------------------------------------
The hoop centre is located at (0, 0, HOOP_HEIGHT).

- x and y define the horizontal plane.
- z is the vertical coordinate.
- Gravity acts along the negative z direction.
- The backboard is represented by a vertical plane at x = BOARD_X.
- The backboard extends along the y direction and has finite dimensions.

The basketball is modelled as a rigid sphere with a fixed radius.
Between collisions, its motion follows the equations of projectile
motion under uniform gravitational acceleration.

Air resistance, Magnus forces, spin, ball rotation and other
aerodynamic effects are neglected.


2. RIM COLLISION MODEL
----------------------
The rim is represented as a circular ring centred on the hoop axis,
located at z = HOOP_HEIGHT, with radius RIM_RADIUS.

The metal rim is modelled as a circular tube with radius TUBE_RADIUS.

A rim collision is detected when the minimum distance between the
basketball centre and the circular rim centreline reaches the sum
of the basketball radius and the tube radius.

The contact time is estimated by sampling the ballistic trajectory
over a finite number of subdivisions. Once a possible contact
interval is identified, a bisection method refines the estimated
contact time.

At impact, the collision normal is calculated from the closest point
on the circular rim centreline to the ball centre.

The velocity is updated using a Newtonian coefficient of restitution.
Only the normal component of the velocity is modified; the tangential
component is preserved. Friction and tangential impulse are neglected.

The ball centre is repositioned at the nominal contact distance to
prevent geometric penetration.


3. BACKBOARD COLLISION MODEL
-----------------------------
The backboard is approximated as a zero-thickness vertical plane.

A collision is considered when the ball moves towards the board
(negative x velocity) and its centre reaches the plane offset by
one basketball radius.

The collision is accepted only if the ball overlaps the finite
backboard dimensions along the y and z directions.

The rebound is modelled using a coefficient of restitution applied
to the x component of velocity. The y and z components are unchanged.

This is a simplified planar collision model. It does not explicitly
resolve impacts with the board's edges, corners or thickness, nor
does it model the detailed geometry of the board support structure.

In addition, backboard collision detection is disabled after the
ball has crossed the vertical plane passing through the hoop centre.
This is a modelling choice implemented in the simulation logic.


4. BASKET DETECTION
-------------------
A potential basket crossing is detected by solving the ballistic
equation for the time at which the ball centre reaches the hoop
height.

Only descending crossings are considered.

At this instant, the radial distance of the ball centre from the
hoop axis is compared with the effective opening radius, accounting
for both the rim tube and the basketball radius.

If the ball centre lies within this opening, the shot is classified
as made.

This is a geometric crossing criterion rather than a detailed
simulation of the ball passing through the entire basket opening.
The net and the subsequent motion of the ball are not modelled.


5. TEMPORAL INTEGRATION AND COLLISION HANDLING
-----------------------------------------------
The simulation advances in small time steps.

Within each time step, the code searches for the earliest possible
rim or backboard collision. The ball is first propagated
ballistically to the estimated event time, the corresponding
collision response is applied, and the remaining part of the time
step is processed.

This event-based approach allows collisions to occur between
regular time-step boundaries.

However, collision detection is not exact for every possible
trajectory. In particular, the rim search uses a finite number of
initial subdivisions and may miss very brief grazing contacts.

The maximum number of collision events processed within a single
time step is also limited to four. If this limit is reached, the
remaining time is propagated without further collision detection.


6. MONTE CARLO SAMPLING
-----------------------
The nominal initial position and velocity are specified by the user.

For each simulated shot, independent Gaussian perturbations are
added to the x, y and z components of the initial velocity.

The standard deviations of these perturbations are controlled by
VELOCITY_SIGMA.

The initial shooting position remains fixed throughout the Monte
Carlo experiment.

Consequently, the resulting distributions describe the effect of
launch-velocity variability under the assumptions of this model.
They do not represent the full variability of real basketball
shooting, which may also involve changes in launch position, spin,
release angle and other factors.


7. REBOUND ANALYSIS
------------------
The rebound analysis retains only shots satisfying all of the
following conditions:

- The shot is classified as missed.
- A valid landing point has been recorded.
- At least one collision with the rim or backboard has occurred.

Three complementary representations are produced:

- An XY map showing horizontal landing positions, the hoop,
  the backboard and the shooter location.
- A radial probability density estimate based on the distance
  between each landing point and the hoop centre.
- A circular angular density estimate using a von Mises kernel,
  describing the distribution of rebound directions.

The radial KDE uses a Gaussian kernel, whereas the angular KDE
uses a circular von Mises kernel to account for the periodic
nature of angles.


8. LANDING POSITION AND NUMERICAL LIMITATIONS
---------------------------------------------
Ground contact is detected at the end of a time step when the ball
centre reaches or passes the basketball radius above the ground
while moving downwards.

The vertical coordinate of the recorded landing point is then set
to the basketball radius.

The horizontal coordinates are taken from the position at the end
of that time step. Therefore, the recorded landing position is an
approximation of the exact ground-contact position.

Similarly, the recorded flight time corresponds to the end of the
time step in which ground contact is detected, rather than the
exact instant of impact with the ground.

These approximations are retained in the present implementation.


9. MODEL LIMITATIONS
--------------------
This simulation is intended as a simplified numerical model for
studying shot outcomes and rebound distributions, rather than as
a fully realistic basketball simulator.

Its main limitations include:

- No air resistance, spin, Magnus effect or aerodynamic drag.
- No friction or tangential impulse during collisions.
- Simplified rim and backboard geometry.
- No explicit modelling of the backboard's thickness, edges,
  corners or support structure.
- Simplified collision detection, which can miss grazing contacts.
- Backboard collisions are disabled after hoop-plane crossing.
- No detailed modelling of the ball's interaction with the net.
- No bouncing or rolling after ground contact.
- Approximate landing coordinates and flight times.
- Independent Gaussian perturbations of initial velocity,
  with no correlations between components.
- No variability in the shooter's position or release mechanics.

The results should therefore be interpreted in the context of
these assumptions. In particular, the predicted landing positions
and rebound distributions depend on the chosen collision parameters
and numerical approximations.
"""


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.special import i0


# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

# Gravitational acceleration (m/s²)
G = 9.81

# Basketball and rim geometry (m)
BALL_RADIUS = 0.12
RIM_RADIUS = 0.2286
TUBE_RADIUS = 0.010
HOOP_HEIGHT = 3.05

# Coefficients of restitution
RESTITUTION_RIM = 0.75
RESTITUTION_BOARD = 0.70

# Backboard geometry (m)
BOARD_X = -RIM_RADIUS  # -0.2286 m
BOARD_WIDTH = 1.83
BOARD_BOTTOM = 2.90
BOARD_TOP = 3.95


# ============================================================
# 2. BALLISTIC MOTION
# ============================================================

def ballistic_position(p0, v0, t):
    """Return the ball position after ballistic propagation."""

    p = p0 + v0 * t
    p[2] -= 0.5 * G * t * t

    return p


def ballistic_velocity(v0, t):
    """Return the ball velocity after ballistic propagation."""

    v = v0.copy()
    v[2] -= G * t

    return v


# ============================================================
# 3. RIM COLLISION MODEL
# ============================================================

def rim_distance(position):
    """Compute the minimum distance to the circular rim centreline."""

    rho = np.hypot(position[0], position[1])

    return np.sqrt(
        (rho - RIM_RADIUS) ** 2
        + (position[2] - HOOP_HEIGHT) ** 2
    )


def first_rim_contact(p0, v0, duration, subdivisions=8):
    """
    Estimate the first rim contact time within the remaining interval.

    A finite sampling is used to locate a possible contact interval.
    Bisection then refines the contact time.
    """

    contact_distance = BALL_RADIUS + TUBE_RADIUS

    def distance_at(t):
        return rim_distance(ballistic_position(p0, v0, t))

    d0 = distance_at(0.0)

    if d0 <= contact_distance + 1e-9:
        return None

    times = np.linspace(0.0, duration, subdivisions + 1)
    distances = [distance_at(t) for t in times]

    for i in range(1, len(times)):

        if (
            distances[i - 1] > contact_distance
            and distances[i] <= contact_distance
        ):

            left = times[i - 1]
            right = times[i]

            for _ in range(40):

                middle = 0.5 * (left + right)

                if distance_at(middle) > contact_distance:
                    left = middle
                else:
                    right = middle

            return 0.5 * (left + right)

    return None


def rim_collision_at_contact(position, velocity):
    """
    Apply a frictionless collision response with the circular rim.

    The normal component of velocity is modified according to the
    coefficient of restitution. The tangential component is preserved.
    """

    contact_distance = BALL_RADIUS + TUBE_RADIUS

    rho = np.hypot(position[0], position[1])

    if rho > 1e-12:
        radial = position[:2] / rho
    else:
        radial = np.array([1.0, 0.0])

    closest = np.array([
        RIM_RADIUS * radial[0],
        RIM_RADIUS * radial[1],
        HOOP_HEIGHT
    ])

    delta = position - closest
    distance = np.linalg.norm(delta)

    if distance < 1e-12:
        return position, velocity, False, None

    normal = delta / distance
    normal_speed = np.dot(velocity, normal)

    # Ignore contacts when the ball is already moving away from the rim.
    if normal_speed >= 0:
        return position, velocity, False, None

    velocity_before = velocity.copy()

    # Correct the ball position to the nominal contact distance.
    position = closest + contact_distance * normal

    # Newtonian restitution law (frictionless collision).
    velocity = (
        velocity
        - (1.0 + RESTITUTION_RIM) * normal_speed * normal
    )

    diagnostic = {
        "velocity_before": velocity_before,
        "velocity_after": velocity.copy(),
        "normal": normal.copy(),
        "closest_point": closest.copy(),
        "contact_position": position.copy()
    }

    return position, velocity, True, diagnostic


# ============================================================
# 4. BACKBOARD COLLISION MODEL
# ============================================================

def first_board_contact(p0, v0, duration):
    """
    Estimate the time of contact with the backboard plane.

    The ball must move towards the board and overlap its finite
    dimensions along the y and z directions.
    """

    contact_x = BOARD_X + BALL_RADIUS

    if v0[0] >= 0:
        return None

    t = (contact_x - p0[0]) / v0[0]

    if t < 0 or t > duration:
        return None

    p = ballistic_position(p0, v0, t)

    overlaps = (
        abs(p[1]) <= BOARD_WIDTH / 2 + BALL_RADIUS
        and p[2] + BALL_RADIUS >= BOARD_BOTTOM
        and p[2] - BALL_RADIUS <= BOARD_TOP
    )

    if not overlaps:
        return None

    return t


def board_collision_at_contact(position, velocity):
    """
    Apply a simplified frictionless collision with the backboard.

    Only the x component of velocity is modified.
    """

    velocity_before = velocity.copy()
    velocity = velocity.copy()

    velocity[0] = -RESTITUTION_BOARD * velocity[0]

    diagnostic = {
        "velocity_before": velocity_before,
        "velocity_after": velocity.copy(),
        "contact_position": position.copy()
    }

    return position, velocity, diagnostic


# ============================================================
# 5. BASKET CROSSING DETECTION
# ============================================================

def detect_basket_crossing(p0, v0, duration):
    """
    Detect a descending crossing of the horizontal hoop plane.

    The ball centre must lie within the effective opening radius
    for the shot to be classified as made.
    """

    z0 = p0[2] - HOOP_HEIGHT
    vz = v0[2]

    discriminant = vz * vz + 2.0 * G * z0

    if discriminant < 0:
        return None

    roots = [
        (vz + np.sqrt(discriminant)) / G,
        (vz - np.sqrt(discriminant)) / G
    ]

    opening_radius = RIM_RADIUS - TUBE_RADIUS - BALL_RADIUS

    for t in sorted(roots):

        if t < 0 or t > duration:
            continue

        if vz - G * t >= 0:
            continue

        crossing = ballistic_position(p0, v0, t)

        radial_distance = np.hypot(
            crossing[0],
            crossing[1]
        )

        return (
            t,
            crossing,
            radial_distance <= opening_radius
        )

    return None


# ============================================================
# 6. SINGLE-SHOT SIMULATION
# ============================================================

def simulate_shot(position_0, velocity_0, dt=0.0002,
                  max_time=3.0, save_trajectory=True):
    """
    Simulate one basketball shot.

    Ballistic motion is interrupted whenever a rim or backboard
    collision is detected. The earliest event is processed first.

    The function returns the shot outcome, collision events,
    landing point, flight time and optionally the trajectory.
    """

    pos = np.array(position_0, dtype=float).copy()
    vel = np.array(velocity_0, dtype=float).copy()

    trajectory = [pos.copy()] if save_trajectory else None
    events = []

    made = False
    crossed_hoop_plane = False
    landing_point = None
    flight_time = 0.0

    n_steps = int(max_time / dt)

    for step in range(n_steps):

        t_step = step * dt
        remaining = dt
        elapsed = 0.0
        contacts_this_step = 0

        while remaining > 1e-12:

            crossing_result = None

            if not crossed_hoop_plane:
                crossing_result = detect_basket_crossing(
                    pos, vel, remaining
                )

            t_rim = first_rim_contact(pos, vel, remaining)

            t_board = first_board_contact(pos, vel, remaining)

            # Backboard collisions are disabled after hoop-plane crossing.
            if crossed_hoop_plane:
                t_board = None

            candidates = []

            if t_rim is not None:
                candidates.append((t_rim, "rim"))

            if t_board is not None:
                candidates.append((t_board, "board"))

            if candidates:
                contact_time, contact_type = min(
                    candidates, key=lambda x: x[0]
                )
            else:
                contact_time = remaining
                contact_type = None

            # Register a basket crossing if it occurs before the next collision.
            if crossing_result is not None:

                crossing_time, crossing_position, in_opening = (
                    crossing_result
                )

                if crossing_time <= contact_time + 1e-12:

                    crossed_hoop_plane = True

                    if in_opening:
                        made = True

                    events.append({
                        "type": "basket" if in_opening else "hoop_plane",
                        "time": t_step + elapsed + crossing_time,
                        "position": crossing_position.copy()
                    })

            # Propagate the ball to the next event.
            pos = ballistic_position(pos, vel, contact_time)
            vel = ballistic_velocity(vel, contact_time)

            elapsed += contact_time
            remaining -= contact_time

            if contact_type is None:
                remaining = 0.0
                break

            absolute_time = t_step + elapsed

            # Apply the corresponding collision response.
            if contact_type == "rim":

                pos, vel, hit, diagnostic = (
                    rim_collision_at_contact(pos, vel)
                )

                if hit:
                    events.append({
                        "type": "rim",
                        "time": absolute_time,
                        "position": pos.copy(),
                        "diagnostic": diagnostic
                    })

            elif contact_type == "board":

                pos, vel, diagnostic = (
                    board_collision_at_contact(pos, vel)
                )

                events.append({
                    "type": "board",
                    "time": absolute_time,
                    "position": pos.copy(),
                    "diagnostic": diagnostic
                })

            contacts_this_step += 1

            # Safety limit on the number of collisions per time step.
            if contacts_this_step >= 4:
                pos = ballistic_position(pos, vel, remaining)
                vel = ballistic_velocity(vel, remaining)
                remaining = 0.0

        if save_trajectory:
            trajectory.append(pos.copy())

        flight_time = t_step + dt

        # Ground-contact detection.
        if pos[2] <= BALL_RADIUS and vel[2] < 0:

            landing_point = pos.copy()
            landing_point[2] = BALL_RADIUS

            break

    return {
        "outcome": "made" if made else "missed",
        "events": events,
        "landing_point": landing_point,
        "flight_time": flight_time,
        "trajectory": (
            np.array(trajectory) if save_trajectory else None
        )
    }


# ============================================================
# 7. MONTE CARLO SIMULATION
# ============================================================

# Random number generator (fixed seed for reproducibility).
rng = np.random.default_rng(42)

# Number of simulated shots.
N = 50

# Shooting angle measured from the hoop-to-shooter x-axis.
angle = np.radians(60)

# Initial shooter position (7 m from the hoop).
position_0 = np.array([
    7.0 * np.cos(angle),
    7.0 * np.sin(angle),
    2.0
])

# Nominal initial velocity (m/s).
velocity_0 = np.array([
    -5.385 * np.cos(angle),
    -5.385 * np.sin(angle),
    7.184
])

# Standard deviations of the Gaussian velocity perturbations (m/s).
velocity_sigma = np.array([0.15, 0.15, 0.15])

# Storage for simulation results.
landing_points = []
outcomes = []

rim_contacts = 0
board_contacts = 0

shot_results = []

for i in tqdm(range(N), desc="Monte Carlo simulation", unit="shot"):

    # Generate random perturbations of the initial velocity.
    velocity = velocity_0 + rng.normal(
        loc=0.0,
        scale=velocity_sigma
    )

    result = simulate_shot(
        position_0,
        velocity,
        dt=0.0002,
        save_trajectory=False
    )

    shot_results.append(result)

    outcomes.append(result["outcome"])

    # Count rim and backboard collision events.
    rim_contacts += sum(
        event["type"] == "rim"
        for event in result["events"]
    )

    board_contacts += sum(
        event["type"] == "board"
        for event in result["events"]
    )

    if result["landing_point"] is not None:
        landing_points.append(result["landing_point"])


landing_points = np.array(landing_points)
outcomes = np.array(outcomes)


# ============================================================
# 8. MONTE CARLO STATISTICS
# ============================================================

n_made = np.sum(outcomes == "made")
n_missed = np.sum(outcomes == "missed")

print("\n===== MONTE CARLO RESULTS =====")
print(f"Number of shots       : {N}")
print(f"Made shots            : {n_made}")
print(f"Missed shots          : {n_missed}")
print(f"Rim contacts          : {rim_contacts}")
print(f"Backboard contacts    : {board_contacts}")

if len(landing_points) > 0:

    print(f"Recorded landing points : {len(landing_points)}")

    print(
        "Mean landing position (x, y) :",
        np.round(landing_points[:, :2].mean(axis=0), 3)
    )


# ============================================================
# 9. REBOUND ANALYSIS
# ============================================================

# ------------------------------------------------------------
# 9.1 Extract valid rebound landing positions
# ------------------------------------------------------------

rebound_points = []

for result in shot_results:

    event_types = {
        event["type"] for event in result["events"]
    }

    landing = result["landing_point"]

    # Retain missed shots with a valid landing point and at least
    # one rim or backboard collision.
    if (
        result["outcome"] == "missed"
        and landing is not None
        and ("rim" in event_types or "board" in event_types)
    ):

        rebound_points.append(
            np.asarray(landing, dtype=float)[:2]
        )


rebound_points = np.asarray(
    rebound_points, dtype=float
).reshape(-1, 2)


print("\n===== REBOUND ANALYSIS =====")
print(f"Rebounds retained : {len(rebound_points)}")


if len(rebound_points) == 0:

    print("No valid rebounds available for analysis.")

else:

    # --------------------------------------------------------
    # 9.2 Compute radial and angular coordinates
    # --------------------------------------------------------

    x_rebounds = rebound_points[:, 0]
    y_rebounds = rebound_points[:, 1]

    # Distance from the hoop centre.
    radii = np.linalg.norm(rebound_points, axis=1)

    # Angular direction of each landing position.
    rebound_angles = np.arctan2(
        y_rebounds,
        x_rebounds
    )

    # Angular position of the shooter as seen from the hoop.
    shooter_angle = np.degrees(
        np.arctan2(position_0[1], position_0[0])
    )

    angle_degrees = np.degrees(angle)

    print(
        f"Median radial distance : {np.median(radii):.3f} m"
    )

    print(
        f"Mean radial distance   : {np.mean(radii):.3f} m"
    )


    # --------------------------------------------------------
    # 9.3 XY rebound map
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 9))

    # Rebound landing positions.
    ax.scatter(
        x_rebounds,
        y_rebounds,
        s=18,
        alpha=0.45,
        label="Rebound landing positions"
    )

    # Circular hoop representation.
    theta_circle = np.linspace(
        0,
        2 * np.pi,
        300
    )

    ax.plot(
        RIM_RADIUS * np.cos(theta_circle),
        RIM_RADIUS * np.sin(theta_circle),
        color="red",
        linewidth=2,
        label="Hoop"
    )

    # Backboard representation.
    board_y = np.linspace(
        -BOARD_WIDTH / 2,
        BOARD_WIDTH / 2,
        100
    )

    ax.plot(
        np.full_like(board_y, BOARD_X),
        board_y,
        color="orange",
        linewidth=4,
        label="Backboard"
    )

    

    # Shooter position.
    ax.scatter(
        position_0[0],
        position_0[1],
        marker="^",
        s=140,
        color="green",
        label="Shooter"
    )

    ax.set_title(
        f"Rebound landing positions — {angle_degrees:.0f}°"
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(-5, 8)
    ax.set_ylim(-8, 8)

    ax.grid(True, alpha=0.3)

    # Place the legend outside the plotting area.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True
    )

    fig.tight_layout()


    # --------------------------------------------------------
    # 9.4 Radial KDE
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    if len(radii) >= 2 and np.std(radii) > 1e-10:

        radial_grid = np.linspace(
            0,
            np.max(radii) + 0.5,
            500
        )

        radial_kde = gaussian_kde(radii)

        ax.plot(
            radial_grid,
            radial_kde(radial_grid),
            linewidth=2.5,
            label="Radial KDE"
        )

        median_radius = np.median(radii)

        ax.axvline(
            median_radius,
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {median_radius:.2f} m"
        )

    else:

        print("Not enough variation for radial KDE.")

    ax.set_title(
        f"Radial rebound distribution — {angle_degrees:.0f}°"
    )

    ax.set_xlabel("Distance from hoop centre (m)")
    ax.set_ylabel("Probability density")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()


    # --------------------------------------------------------
    # 9.5 Circular angular KDE (von Mises kernel)
    # --------------------------------------------------------

    def circular_kde(angles, grid, kappa=8.0):
        """
        Estimate angular density using a von Mises kernel.

        The von Mises distribution is the circular equivalent
        of a Gaussian distribution. Kappa controls concentration:
        larger values produce narrower kernels.
        """

        differences = grid[:, None] - angles[None, :]

        kernels = (
            np.exp(kappa * np.cos(differences))
            / (2 * np.pi * i0(kappa))
        )

        return kernels.mean(axis=1)


    if len(rebound_angles) > 0:

        angle_grid = np.linspace(
            -np.pi,
            np.pi,
            721
        )

        angular_density = circular_kde(
            rebound_angles,
            angle_grid,
            kappa=8.0
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            np.degrees(angle_grid),
            angular_density,
            linewidth=2.5,
            label="Angular KDE"
        )

        # Shooter angular position.
        ax.axvline(
            shooter_angle,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Shooter direction: {shooter_angle:.1f}°"
        )

        ax.set_title(
            f"Angular rebound distribution — {angle_degrees:.0f}°"
        )

        ax.set_xlabel("Rebound direction (°)")
        ax.set_ylabel("Angular probability density (rad⁻¹)")

        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 45))

        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()



    # Display all generated figures.
    plt.show()
