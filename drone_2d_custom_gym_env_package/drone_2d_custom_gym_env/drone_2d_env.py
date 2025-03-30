# Assuming Drone class is in the same directory or accessible via path
try:
    # If running as part of a package
    from .Drone import Drone
    from .event_handler import pygame_events  # Assuming event_handler is also relative
except ImportError:
    # If running the script directly
    # Make sure these paths are correct for your project structure
    from drone_2d_custom_gym_env_package.drone_2d_custom_gym_env.Drone import Drone
    from drone_2d_custom_gym_env_package.drone_2d_custom_gym_env.event_handler import (
        pygame_events,
    )

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import os
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import math  # Import math for atan2


class Drone2dEnv(gym.Env):
    metadata = {
        "render.modes": ["human"],
        "render_fps": 50,
        "render.options": ["render_path", "render_shade", "shade_distance_m"],
    }

    
    def __init__(
        self,
        render_sim: bool = False,              # If True, initializes Pygame and enables visual rendering. Set to False for faster training.
        max_steps: int = 1000,                 # Maximum number of simulation steps allowed per episode before termination.
        render_path: bool = True,              # If True (and render_sim=True), draws the drone's trajectory history.
        render_shade: bool = True,             # If True (and render_sim=True), renders faded drone images ("shades") along the path.
        shade_distance_m: float = 2.0,         # Distance (meters) the drone must travel before a new shade is rendered.
        moving_platform: bool = False,         # If True, the landing platform will move horizontally.
        platform_speed: float = 1.5,           # Horizontal speed (m/s) of the platform IF moving_platform is True.
        initial_pos_random_range_m: float = 5.0,# Half-width/height (meters) of the square zone for randomizing the drone's starting position. Set to 0 for fixed start.
        max_allowed_tilt_angle_rad: float = np.pi / 2.0, # Maximum absolute tilt angle (radians, from vertical) before 'lost_control' termination (default: 90 degrees).
        enable_wind: bool = True,              # If True, applies wind forces during the simulation.
        reward_landing: float = 100.0,         # Reward for landing if is incresed increase the interest to do it max 500.0 
        reward_un_landing: float = 20.0       # Reward for unstable landing if is incresed increase the interest to do it max 100.0 
    ):
        """
        Initializes the Drone Landing Environment.

        Args:
            render_sim: Enable visual rendering via Pygame.
            max_steps: Maximum steps per episode.
            render_path: Draw drone trajectory.
            render_shade: Draw drone shades.
            shade_distance_m: Distance between shades in meters.
            moving_platform: Enable horizontal platform movement.
            platform_speed: Speed of the platform if moving (m/s).
            initial_pos_random_range_m: Half-range for randomizing start position (m).
            max_allowed_tilt_angle_rad: Max tilt angle before termination (radians).
            enable_wind: Enable wind simulation.
        """
        super().__init__()

        # --- Store Initialization Parameters ---
        self.render_sim = render_sim
        self.max_steps = max_steps
        self.render_path = render_path
        self.render_shade = render_shade
        self.shade_distance_m = shade_distance_m
        self.moving_platform = moving_platform
        self.platform_speed = platform_speed if moving_platform else 0.0 # Store actual speed based on flag
        self.initial_pos_random_range_m = initial_pos_random_range_m
        self.max_allowed_tilt_angle_rad = max_allowed_tilt_angle_rad
        self.enable_wind = enable_wind
        self.reward_landing = reward_landing
        self.reward_un_landing = reward_un_landing

        # === Platform Specific State ===
        self.platform_direction = 1 # Initial direction (1=right, -1=left), randomized in reset if moving
        self.pad_body = None # Reference to the Pymunk body of the landing pad

        # === Environment Physics Parameters ===
        self.gravity_mag: float = 9.81                  # Acceleration due to gravity (m/s^2)
        self.wind_speed: float = 5.0 if self.enable_wind else 0.0 # Actual wind speed used (m/s), depends on enable_wind flag
        self.wind_force_coefficient: float = 0.5        # Simple factor multiplied by wind speed and drone width to calculate wind force magnitude. Tune based on desired wind effect strength.
        # self.wind_direction is initialized in seed() using np_random for reproducibility

        # === Drone Physical Parameters ===
        self.lander_mass: float = 1.0                   # Total mass of the drone (kg)
        self.lander_width: float = 1.0                  # Total width of the drone (meters, e.g., motor tip to motor tip)
        self.lander_height: float = 0.2                 # Height/diameter of the motor components (meters), used for visual representation and joint placement.
        self.initial_Battery: float = 100.0             # Starting amount of battery charge (arbitrary units)
        self.max_thrust: float = 15.0                   # Maximum thrust force (Newtons) EACH motor can produce. Should be > (mass * g / 2) to hover.
        self.thrust_noise_std_dev: float = 0.05         # Standard deviation of Gaussian noise added to thrust commands (as a fraction of max_thrust). Adds realism/robustness.
        self.Battery_consumption_rate: float = 0.1      # Units of battery consumed per Newton-second of total thrust applied. Controls flight time.

        # === World Dimensions ===
        self.world_width: float = 50.0                  # Width of the simulation area (meters)
        self.world_height: float = 50.0                 # Height of the simulation area (meters)
        self.ground_height: float = 10.0                # Vertical position (Y-coordinate) of the ground surface (meters)

        # === Landing Task Parameters ===
        self.landing_pad_width: float = 5.0             # Width of the landing platform (meters)
        self.landing_pad_height: float = 0.5            # Thickness of the landing platform (meters)
        self.initial_landing_target_x: float = self.world_width / 2.0 # Initial X-coordinate of the landing platform's center (meters). Platform resets here if static.
        self.landing_target_y: float = self.ground_height # Y-coordinate of the BOTTOM of the landing platform (meters). It sits on the ground.
        self.max_safe_landing_speed: float = 1.5        # Maximum impact velocity magnitude (m/s) allowed for a safe landing.
        self.max_safe_landing_angle: float = 0.2        # Maximum absolute tilt angle (radians, approx 11 deg) allowed from vertical for a safe landing.

        # === Simulation Timing ===
        self.frames_per_second: int = 50                # Target physics update rate (Hz). Also used for default rendering FPS.
        self.dt: float = 1.0 / self.frames_per_second   # Simulation time step (seconds).

        # === Internal Simulation State (Reset in self.reset()) ===
        self.current_step: int = 0                      # Counter for steps within the current episode.
        self.landed_safely: bool = False                # Flag indicating successful landing in the current episode.
        self.crashed: bool = False                      # Flag indicating a crash (ground or hard pad collision) in the current episode.
        self.Battery_empty: bool = False                # Flag indicating the battery ran out in the current episode.
        self.out_of_bounds: bool = False                # Flag indicating the drone went outside world boundaries.
        self.lost_control: bool = False                 # Flag indicating the drone exceeded the tilt angle limit.
        self.current_left_thrust_applied: float = 0.0   # Actual thrust applied by the left motor in the last step (for rendering).
        self.current_right_thrust_applied: float = 0.0  # Actual thrust applied by the right motor in the last step (for rendering).
        self.info: dict = {}                            # Dictionary for auxiliary diagnostic information returned by step().

        # === Rendering Specific State (Reset in self.reset()) ===
        self.screen_width_px: int = 800                 # Width of the Pygame window in pixels.
        self.screen_height_px: int = 800                # Height of the Pygame window in pixels.
        self.pixels_per_meter: float = min(self.screen_width_px / self.world_width, self.screen_height_px / self.world_height) # Scale factor for rendering meters->pixels.
        self.screen: pygame.Surface | None = None       # Pygame display surface object.
        self.clock: pygame.time.Clock | None = None     # Pygame clock object for controlling FPS.
        self.font: pygame.font.Font | None = None       # Pygame font object for rendering text.
        self.shade_image: pygame.Surface | None = None  # Loaded and scaled Pygame surface for the drone shade image.
        self.flight_path_px: list = []                  # List storing historical drone positions in PIXEL coordinates for path rendering.
        self.path_drone_shade_info: list = []           # List storing [world_x, world_y, angle_rad] for each rendered shade.
        self.last_shade_pos: Vec2d | None = None        # World position (Vec2d) where the last shade was dropped.

        # Initialize Pygame if rendering is enabled
        if self.render_sim:
            self.init_pygame()

        # === Action & Observation Spaces (Gym Interface) ===
        min_action = np.array([-1.0, -1.0], dtype=np.float32)
        max_action = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32) # Continuous actions for left/right thrust [-1, 1]

        obs_dim = 11 # Base + Sensor info
        if self.moving_platform: obs_dim = 12 # Add platform velocity
        obs_low_list = [-1,-1,-1,-1,-1,-1,0,-1,-1,-1,0.0]; obs_high_list = [1,1,1,1,1,1,1,1,1,1,1.0]
        if self.moving_platform: obs_low_list.append(-1); obs_high_list.append(1)
        obs_low = np.array(obs_low_list, dtype=np.float32); obs_high = np.array(obs_high_list, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32) # Define bounds and shape of the observation vector
        print(f"Observation Space Size: {self.observation_space.shape}") # Informative print

        # --- Initialize Physics Space & Objects ---
        self.space: pymunk.Space | None = None          # Pymunk physics space object.
        self.drone: Drone | None = None                 # Instance of the Drone class.
        self.landing_pad_shape: pymunk.Shape | None = None # Pymunk shape for the landing pad.
        self.ground_shape: pymunk.Shape | None = None   # Pymunk shape for the ground.

        # Initialize random number generator and perform initial reset
        self.seed()
        self.reset() # Creates space, drone, pad, etc. and returns initial observation

    # seed, _world_to_screen, _screen_to_world, init_pygame, init_pymunk, _add_position_to_path, _add_drone_shade, collision_begin, collision_separate methods remain the same...
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.wind_direction = self.np_random.uniform(0, 2 * np.pi)
        if self.moving_platform:
            self.platform_direction = self.np_random.choice([-1, 1])
            return [seed]

    def _world_to_screen(self, world_pos):
        wx, wy = world_pos
        sx = wx * self.pixels_per_meter
        sy = self.screen_height_px - (wy * self.pixels_per_meter)
        return int(sx), int(sy)

    def _screen_to_world(self, screen_pos):
        sx, sy = screen_pos
        wx = sx / self.pixels_per_meter
        wy = (self.screen_height_px - sy) / self.pixels_per_meter
        return wx, wy

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.screen_width_px, self.screen_height_px)
        )
        pygame.display.set_caption("Drone Landing Environment (Metric)")
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Arial", 16)
        except Exception:
            self.font = pygame.font.Font(None, 20)
        if self.render_shade:
            try:
                script_dir = os.path.dirname(__file__)
                img_path = os.path.join(script_dir, "img", "shade.png")
                original_shade_image = pygame.image.load(img_path).convert_alpha()
            except Exception as e:
                print(f"Warning: Error loading shade image {img_path}: {e}.")
                original_shade_image = None
            if original_shade_image:
                try:
                    target_width_px = int(self.lander_width * self.pixels_per_meter)
                    orig_w, orig_h = original_shade_image.get_size()
                    if orig_w > 0:
                        aspect_ratio = orig_h / orig_w
                        target_height_px = int(target_width_px * aspect_ratio)
                    else:
                        target_height_px = int(
                            self.lander_height * self.pixels_per_meter
                        )
                    if target_width_px > 0 and target_height_px > 0:
                        self.shade_image = pygame.transform.smoothscale(
                            original_shade_image, (target_width_px, target_height_px)
                        )
                    else:
                        self.shade_image = original_shade_image
                except Exception as e:
                    print(f"Warning: Error scaling shade image: {e}.")
                    self.shade_image = original_shade_image
            else:
                self.shade_image = None

    def init_pymunk(self):
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -self.gravity_mag)
        self.drone_collision_type = 1
        self.pad_collision_type = 2
        self.ground_collision_type = 3
        ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.ground_shape = pymunk.Segment(
            ground_body,
            (0, self.ground_height),
            (self.world_width, self.ground_height),
            0.5,
        )
        self.ground_shape.friction = 0.8
        self.ground_shape.collision_type = self.ground_collision_type
        self.space.add(ground_body, self.ground_shape)
        pad_body_type = (
            pymunk.Body.KINEMATIC if self.moving_platform else pymunk.Body.STATIC
        )
        self.pad_body = pymunk.Body(body_type=pad_body_type)
        self.pad_body.position = (self.initial_landing_target_x, self.landing_target_y)
        self.pad_body.velocity = (0, 0)
        half_w = self.landing_pad_width / 2.0
        pad_verts = [
            (-half_w, 0),
            (half_w, 0),
            (half_w, self.landing_pad_height),
            (-half_w, self.landing_pad_height),
        ]
        self.landing_pad_shape = pymunk.Poly(self.pad_body, pad_verts)
        self.landing_pad_shape.friction = 1.0
        self.landing_pad_shape.collision_type = self.pad_collision_type
        self.space.add(self.pad_body, self.landing_pad_shape)
        center_x = self.world_width / 2.0
        center_y = self.world_height * 0.8
        r = self.initial_pos_random_range_m
        initial_x = self.np_random.uniform(center_x - r, center_x + r)
        initial_y = self.np_random.uniform(center_y - r, center_y + r)
        drone_half_width = self.lander_width / 2.0
        initial_x = np.clip(
            initial_x, drone_half_width, self.world_width - drone_half_width
        )
        min_start_y = self.ground_height + self.landing_pad_height + self.lander_height
        initial_y = np.clip(
            initial_y, min_start_y, self.world_height - self.lander_height
        )
        initial_angle = self.np_random.uniform(-0.1, 0.1)
        self.drone = Drone(
            initial_x,
            initial_y,
            initial_angle,
            self.lander_height,
            self.lander_width,
            self.lander_mass,
            self.space,
        )
        print(f"Drone starting at: ({initial_x:.2f}, {initial_y:.2f}) m")
        for shape in self.drone.shapes:
            shape.collision_type = self.drone_collision_type
            shape.friction = 0.7
        handler_pad = self.space.add_collision_handler(
            self.drone_collision_type, self.pad_collision_type
        )
        handler_pad.begin = self.collision_begin
        handler_pad.separate = self.collision_separate
        handler_ground = self.space.add_collision_handler(
            self.drone_collision_type, self.ground_collision_type
        )
        handler_ground.begin = self.collision_begin
        handler_ground.separate = self.collision_separate
        self.drone_contacts = 0

    def _add_position_to_path(self):
        if not self.render_sim or not self.render_path or self.drone is None:
            return
        pos_screen = self._world_to_screen(self.drone.body.position)
        self.flight_path_px.append(pos_screen)

    def _add_drone_shade(self):
        if not self.render_sim or not self.render_shade or self.drone is None:
            return
        pos_world = self.drone.body.position
        angle_rad = self.drone.body.angle
        self.path_drone_shade_info.append([pos_world.x, pos_world.y, angle_rad])
        self.last_shade_pos = pos_world

    def collision_begin(self, arbiter, space, data):
        self.drone_contacts += 1
        is_pad_collision = (
            arbiter.shapes[0].collision_type == self.pad_collision_type
            or arbiter.shapes[1].collision_type == self.pad_collision_type
        )
        drone_body = self.drone.body
        velocity = drone_body.velocity
        angle = abs(drone_body.angle % (2 * np.pi))
        if angle > np.pi:
            angle -= 2 * np.pi
            angle = abs(angle)
        if is_pad_collision:
            if (
                velocity.length < self.max_safe_landing_speed
                and angle < self.max_safe_landing_angle
            ):
                self.landed_safely = True
                print(f"Step {self.current_step}: LANDED SAFELY!")
            else:
                self.crashed = True
                print(
                    f"Step {self.current_step}: CRASHED on pad! Speed: {velocity.length:.2f}, Angle: {angle:.2f}"
                )
        else:
            self.crashed = True
            print(f"Step {self.current_step}: CRASHED on ground!")
        return True

    def collision_separate(self, arbiter, space, data):
        self.drone_contacts -= 1
        self.drone_contacts = max(0, self.drone_contacts)

    # MODIFIED _apply_forces
    def _apply_forces(self, action):
        left_thrust_cmd = (action[0] + 1.0) / 2.0 * self.max_thrust
        right_thrust_cmd = (action[1] + 1.0) / 2.0 * self.max_thrust
        left_noise = self.np_random.normal(
            0, self.thrust_noise_std_dev * self.max_thrust
        )
        right_noise = self.np_random.normal(
            0, self.thrust_noise_std_dev * self.max_thrust
        )
        left_thrust = np.clip(left_thrust_cmd + left_noise, 0, self.max_thrust)
        right_thrust = np.clip(right_thrust_cmd + right_noise, 0, self.max_thrust)
        applied_left_thrust = left_thrust if self.current_Battery > 0 else 0
        applied_right_thrust = right_thrust if self.current_Battery > 0 else 0

        # Apply thrust
        if self.drone:
            self.drone.apply_thrust(applied_left_thrust, applied_right_thrust)

        # Store applied thrust for rendering
        self.current_left_thrust_applied = applied_left_thrust
        self.current_right_thrust_applied = applied_right_thrust

        # Consume Battery
        Battery_consumed = (
            (applied_left_thrust + applied_right_thrust)
            * self.Battery_consumption_rate
            * self.dt
        )
        self.current_Battery -= Battery_consumed
        if self.current_Battery <= 0:
            self.current_Battery = 0
        if not self.Battery_empty and self.current_Battery == 0:
            print(f"Step {self.current_step}: Battery EMPTY!")
            self.Battery_empty = True

        # Apply Wind only if enabled - MODIFIED
        if self.enable_wind and self.wind_speed > 0:
            wind_force_vec = Vec2d(self.wind_speed, 0).rotated(self.wind_direction)
            wind_force = (
                wind_force_vec * self.wind_force_coefficient * self.lander_width
            )
            if self.drone:
                self.drone.body.apply_force_at_world_point(
                    wind_force, self.drone.body.position
                )

    # _get_observation, _calculate_reward, _check_termination, _update_platform_position, step methods remain the same...
    def _get_observation(self):
        platform_pos_x = self.initial_landing_target_x
        platform_target_y = self.landing_target_y + self.landing_pad_height
        platform_vel_x = 0.0
        if self.pad_body:
            platform_pos_x = self.pad_body.position.x
        if self.pad_body and self.moving_platform:
            platform_vel_x = self.pad_body.velocity.x
        if self.drone is None or self.drone.body is None:
            return np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )
        drone_body = self.drone.body
        pos = drone_body.position
        vel = drone_body.velocity
        angle = drone_body.angle
        angular_vel = drone_body.angular_velocity
        world_vector_to_target = Vec2d(
            platform_pos_x - pos.x, platform_target_y - pos.y
        )
        distance_to_target = world_vector_to_target.length
        world_angle_to_target = math.atan2(
            world_vector_to_target.y, world_vector_to_target.x
        )
        relative_angle_to_target = world_angle_to_target - angle
        relative_angle_to_target = (relative_angle_to_target + np.pi) % (
            2 * np.pi
        ) - np.pi
        err_x = world_vector_to_target.x
        err_y = world_vector_to_target.y
        norm_pos_x = np.clip(pos.x / self.world_width * 2 - 1, -1.0, 1.0)
        norm_pos_y = np.clip(pos.y / self.world_height * 2 - 1, -1.0, 1.0)
        norm_vel_x = np.clip(vel.x / 15.0, -1.0, 1.0)
        norm_vel_y = np.clip(vel.y / 15.0, -1.0, 1.0)
        norm_angle = (angle + np.pi) % (2 * np.pi) - np.pi
        norm_angle = np.clip(norm_angle / np.pi, -1.0, 1.0)
        norm_angular_vel = np.clip(angular_vel / 10.0, -1.0, 1.0)
        norm_Battery = self.current_Battery / self.initial_Battery
        norm_err_x = np.clip(err_x / (self.world_width / 2.0), -1.0, 1.0)
        norm_err_y = np.clip(err_y / (self.world_height / 2.0), -1.0, 1.0)
        norm_relative_angle = np.clip(relative_angle_to_target / np.pi, -1.0, 1.0)
        norm_distance = np.clip(distance_to_target / self.world_height, 0.0, 1.0)
        obs_list = [
            norm_pos_x,
            norm_pos_y,
            norm_vel_x,
            norm_vel_y,
            norm_angle,
            norm_angular_vel,
            norm_Battery,
            norm_err_x,
            norm_err_y,
            norm_relative_angle,
            norm_distance,
        ]
        if self.moving_platform:
            norm_platform_vx = np.clip(
                platform_vel_x / (self.platform_speed + 1e-6), -1.0, 1.0
            )
            obs_list.append(norm_platform_vx)
        obs = np.array(obs_list, dtype=np.float32)
        if np.isnan(obs).any() or np.isinf(obs).any():
            print("!!! WARNING: NaN or Inf detected in observation !!!")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            self.crashed = True
        return obs

    def _calculate_reward(self, observation):
        expected_len = 11
        if self.moving_platform:
            expected_len = 12
        if len(observation) != expected_len:
            print(
                f"!!! WARNING: Obs length mismatch. Expected {expected_len}, got {len(observation)} !!!"
            )
            return 0.0
        if self.moving_platform:
            (
                norm_pos_x,
                norm_pos_y,
                norm_vel_x,
                norm_vel_y,
                norm_angle,
                norm_angular_vel,
                norm_Battery,
                norm_err_x,
                norm_err_y,
                norm_relative_angle,
                norm_distance,
                norm_platform_vx,
            ) = observation
        else:
            (
                norm_pos_x,
                norm_pos_y,
                norm_vel_x,
                norm_vel_y,
                norm_angle,
                norm_angular_vel,
                norm_Battery,
                norm_err_x,
                norm_err_y,
                norm_relative_angle,
                norm_distance,
            ) = observation
        reward = 0.0
        err_x = norm_err_x * (self.world_width / 2.0)
        err_y = norm_err_y * (self.world_height / 2.0)
        dist_to_target = np.sqrt(err_x**2 + err_y**2)
        reward += 2.0 * np.exp(-0.1 * dist_to_target)
        velocity_penalty = (norm_vel_x**2 + norm_vel_y**2) * 0.1
        reward -= velocity_penalty
        angle_penalty = abs(norm_angle) * 0.2
        angular_vel_penalty = abs(norm_angular_vel) * 0.1
        reward -= angle_penalty + angular_vel_penalty
        if self.landed_safely:
            drone_body = self.drone.body if self.drone else None
            if (
                drone_body
                and drone_body.velocity.length < 0.1
                and abs(drone_body.angular_velocity) < 0.1
            ):
                reward += self.reward_landing # this reward is for landing 
            else:
                reward += self.reward_un_landing # unstable landing
        elif self.crashed:
            reward -= 50.0
        elif self.lost_control:
            reward -= 50.0
        elif self.Battery_empty and not (self.landed_safely or self.crashed):
            reward -= 50.0
        elif self.out_of_bounds:
            reward -= 50.0
        elif self.current_step >= self.max_steps and not (
            self.landed_safely or self.crashed
        ):
            reward -= 10.0
        return reward

    def _check_termination(self):
        done = False
        drone_pos = None
        if self.drone and self.drone.body:
            drone_pos = self.drone.body.position
            angle_rad = self.drone.body.angle
        else:
            print("Warning: Drone object missing in _check_termination.")
            done = True
            self.crashed = True
        if not done and drone_pos:
            if not (
                0 < drone_pos.x < self.world_width
                and 0 < drone_pos.y < self.world_height
            ):
                if not self.out_of_bounds:
                    print(f"Step {self.current_step}: OUT OF BOUNDS!")
                    self.out_of_bounds = True
                    done = True
            if not done:
                angle_norm = (angle_rad + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_norm) > self.max_allowed_tilt_angle_rad:
                    if not self.lost_control:
                        print(f"Step {self.current_step}: LOST CONTROL!")
                        self.lost_control = True
                        done = True
        if not done and self.Battery_empty and not (self.landed_safely or self.crashed):
            done = True
        if not done and (self.landed_safely or self.crashed):
            done = True
        if not done and self.current_step >= self.max_steps:
            if not (
                self.landed_safely
                or self.crashed
                or self.out_of_bounds
                or self.lost_control
                or self.Battery_empty
            ):
                print(f"Step {self.current_step}: MAX STEPS REACHED!")
            done = True
        return done

    def _update_platform_position(self):
        if not self.moving_platform or self.pad_body is None:
            return
        min_center_x = self.landing_pad_width / 2.0
        max_center_x = self.world_width - (self.landing_pad_width / 2.0)
        current_x = self.pad_body.position.x
        if current_x <= min_center_x and self.platform_direction == -1:
            self.platform_direction = 1
        elif current_x >= max_center_x and self.platform_direction == 1:
            self.platform_direction = -1
        self.pad_body.velocity = Vec2d(self.platform_speed * self.platform_direction, 0)

    def step(self, action):
        if self.drone is None:
            dummy_obs = np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )
            return dummy_obs, 0.0, True, {"error": "Drone not initialized"}
        self._update_platform_position()
        self._apply_forces(action)
        self.space.step(self.dt)
        current_pos = self.drone.body.position
        self._add_position_to_path()
        if self.render_shade and self.last_shade_pos is not None:
            if current_pos.get_distance(self.last_shade_pos) > self.shade_distance_m:
                self._add_drone_shade()
        observation = self._get_observation()
        done = self._check_termination()
        reward = self._calculate_reward(observation)
        self.current_step += 1
        drone_body = self.drone.body
        self.info = {
            "Battery": self.current_Battery,
            "landed": self.landed_safely,
            "crashed": self.crashed,
            "out_of_bounds": self.out_of_bounds,
            "Battery_empty": self.Battery_empty,
            "lost_control": self.lost_control,
            "steps": self.current_step,
            "raw_pos": tuple(drone_body.position),
            "raw_vel": tuple(drone_body.velocity),
            "raw_angle_rad": drone_body.angle,
            "raw_angular_vel": drone_body.angular_velocity,
            "platform_pos_x": (
                self.pad_body.position.x
                if self.pad_body
                else self.initial_landing_target_x
            ),
            "platform_vel_x": self.pad_body.velocity.x if self.pad_body else 0.0,
        }
        return observation, reward, done, self.info

    # reset remains the same
    def reset(self):
        if self.space:
            for body in list(self.space.bodies):
                self.space.remove(body)
            for shape in list(self.space.shapes):
                self.space.remove(shape)
            for constraint in list(self.space.constraints):
                self.space.remove(constraint)
        self.space = None
        self.drone = None
        self.pad_body = None
        self.current_step = 0
        self.landed_safely = False
        self.crashed = False
        self.Battery_empty = False
        self.out_of_bounds = False
        self.lost_control = False
        self.current_Battery = self.initial_Battery
        self.drone_contacts = 0
        self.current_left_thrust_applied = 0.0
        self.current_right_thrust_applied = 0.0
        self.flight_path_px = []
        self.path_drone_shade_info = []
        self.last_shade_pos = None
        self.wind_direction = self.np_random.uniform(0, 2 * np.pi)
        if self.moving_platform:
            self.platform_direction = self.np_random.choice([-1, 1])
        else:
            self.platform_direction = 1
        self.init_pymunk()
        if self.drone:
            self._add_position_to_path()
            self._add_drone_shade()
        else:
            print("Error: Drone not created during reset.")
        # --- MODIFIED Print Statement ---
        print(
            f"Environment Reset. Wind Enabled: {self.enable_wind}, Wind Dir: {np.degrees(self.wind_direction):.1f} deg. Moving Platform: {self.moving_platform}"
        )
        # --- End Modification ---
        if self.drone:
            return self._get_observation()
        else:
            return np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )

    # MODIFIED render
    def render(self, mode="human"):
        if not self.render_sim or self.screen is None:
            return
        if self.drone is None:
            print("Warning: Trying to render but drone is not initialized.")
            return

        try:
            pygame_events(self)
        except NameError:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    import sys

                    sys.exit()

        self.screen.fill((210, 230, 250))  # Background

        # Draw Initial Spawn Zone
        if self.initial_pos_random_range_m > 0:
            center_x_m = self.world_width / 2.0
            center_y_m = self.world_height * 0.8
            r_m = self.initial_pos_random_range_m
            zone_color = (200, 200, 150)
            zone_thickness = 1
            top_left_m = (center_x_m - r_m, center_y_m + r_m)
            bottom_right_m = (center_x_m + r_m, center_y_m - r_m)
            top_left_px = self._world_to_screen(top_left_m)
            bottom_right_px = self._world_to_screen(bottom_right_m)
            rect_width_px = abs(bottom_right_px[0] - top_left_px[0])
            rect_height_px = abs(bottom_right_px[1] - top_left_px[1])
            spawn_rect = pygame.Rect(
                top_left_px[0], top_left_px[1], rect_width_px, rect_height_px
            )
            pygame.draw.rect(self.screen, zone_color, spawn_rect, zone_thickness)

        # Draw Shades
        if self.render_shade and self.shade_image:
            for shade_info in self.path_drone_shade_info:
                try:
                    angle_deg = -np.degrees(shade_info[2])
                    rotated_image = pygame.transform.rotate(self.shade_image, angle_deg)
                    center_px = self._world_to_screen((shade_info[0], shade_info[1]))
                    image_rect = rotated_image.get_rect(center=center_px)
                    self.screen.blit(rotated_image, image_rect)
                except Exception as e:
                    print(f"Error rendering shade: {e}")

        # Draw Paths
        if self.render_path and len(self.flight_path_px) > 1:
            pygame.draw.aalines(self.screen, (16, 19, 97), False, self.flight_path_px)

        # Draw Ground, Pad, Drone...
        ground_start_px = self._world_to_screen((0, self.ground_height))
        ground_end_px = self._world_to_screen((self.world_width, self.ground_height))
        pygame.draw.line(self.screen, (70, 170, 70), ground_start_px, ground_end_px, 4)
        if self.landing_pad_shape and self.pad_body:
            pad_verts_world = [
                self.pad_body.local_to_world(v)
                for v in self.landing_pad_shape.get_vertices()
            ]
            pad_verts_screen = [self._world_to_screen(v) for v in pad_verts_world]
            pygame.draw.polygon(self.screen, (150, 150, 170), pad_verts_screen)
            pygame.draw.polygon(self.screen, (50, 50, 50), pad_verts_screen, 2)
        if self.drone:
            for shape in self.drone.shapes:
                body = shape.body
                if isinstance(shape, pymunk.Poly):
                    verts_world = [body.local_to_world(v) for v in shape.get_vertices()]
                    verts_screen = [self._world_to_screen(v) for v in verts_world]
                    color = shape.color if hasattr(shape, "color") else (100, 100, 200)
                    pygame.draw.polygon(self.screen, color, verts_screen)
                    pygame.draw.polygon(self.screen, (0, 0, 0), verts_screen, 1)

        # Draw Motor Force Vectors
        if self.drone:
            drone_body = self.drone.body
            max_thrust_render_length_px = 40
            grey_color = (170, 170, 170)
            red_color = (255, 50, 50)
            line_width = 3
            max_thrust_render_length_m = (
                max_thrust_render_length_px / self.pixels_per_meter
            )
            max_thrust_val = self.max_thrust if self.max_thrust > 1e-6 else 1.0
            try:  # Left Motor
                left_motor_attach_local = (-self.drone.thrust_arm_length, 0)
                max_thrust_offset_local = Vec2d(0, max_thrust_render_length_m)
                current_thrust_offset_local = max_thrust_offset_local * (
                    self.current_left_thrust_applied / max_thrust_val
                )
                start_world = drone_body.local_to_world(left_motor_attach_local)
                max_end_world = drone_body.local_to_world(
                    left_motor_attach_local + max_thrust_offset_local
                )
                current_end_world = drone_body.local_to_world(
                    left_motor_attach_local + current_thrust_offset_local
                )
                start_screen = self._world_to_screen(start_world)
                max_end_screen = self._world_to_screen(max_end_world)
                current_end_screen = self._world_to_screen(current_end_world)
                pygame.draw.line(
                    self.screen, grey_color, start_screen, max_end_screen, line_width
                )
                pygame.draw.line(
                    self.screen, red_color, start_screen, current_end_screen, line_width
                )
            except Exception as e:
                print(f"Error drawing left thrust vector: {e}")
            try:  # Right Motor
                right_motor_attach_local = (self.drone.thrust_arm_length, 0)
                max_thrust_offset_local = Vec2d(0, max_thrust_render_length_m)
                current_thrust_offset_local = max_thrust_offset_local * (
                    self.current_right_thrust_applied / max_thrust_val
                )
                start_world = drone_body.local_to_world(right_motor_attach_local)
                max_end_world = drone_body.local_to_world(
                    right_motor_attach_local + max_thrust_offset_local
                )
                current_end_world = drone_body.local_to_world(
                    right_motor_attach_local + current_thrust_offset_local
                )
                start_screen = self._world_to_screen(start_world)
                max_end_screen = self._world_to_screen(max_end_world)
                current_end_screen = self._world_to_screen(current_end_world)
                pygame.draw.line(
                    self.screen, grey_color, start_screen, max_end_screen, line_width
                )
                pygame.draw.line(
                    self.screen, red_color, start_screen, current_end_screen, line_width
                )
            except Exception as e:
                print(f"Error drawing right thrust vector: {e}")

        # --- Draw Wind Indicator and Label --- MODIFIED ---
        if self.enable_wind and self.wind_speed > 0:  # Only draw if wind is enabled
            wind_indicator_base_pos = (self.screen_width_px - 70, 50)
            wind_color = (50, 50, 200)
            wind_label_color = (0, 0, 0)
            arrow_length_px = min(self.wind_speed * self.pixels_per_meter * 3, 50)
            wind_vec_screen = Vec2d(arrow_length_px, 0).rotated(-self.wind_direction)
            wind_end_px = (
                wind_indicator_base_pos[0] + wind_vec_screen.x,
                wind_indicator_base_pos[1] + wind_vec_screen.y,
            )
            pygame.draw.line(
                self.screen, wind_color, wind_indicator_base_pos, wind_end_px, 3
            )
            pygame.draw.circle(self.screen, wind_color, wind_indicator_base_pos, 5)
            if self.font:
                wind_dir_degrees = np.degrees(self.wind_direction) % 360
                wind_label_text = (
                    f"Wind: {self.wind_speed:.1f} m/s @ {wind_dir_degrees:.0f}°"
                )
                wind_label_surf = self.font.render(
                    wind_label_text, True, wind_label_color
                )
                label_pos = (
                    wind_indicator_base_pos[0] - wind_label_surf.get_width() - 10,
                    wind_indicator_base_pos[1] + 5,
                )
                self.screen.blit(wind_label_surf, label_pos)
        # --- End Wind Indicator ---

        # Draw Info Text
        if self.font:
            start_y = 10
            line_height = 20
            text_color = (0, 0, 0)
            status_color_ok = (0, 150, 0)
            status_color_bad = (200, 0, 0)
            battery_text = (
                f"Battery: {self.current_Battery:.1f} / {self.initial_Battery:.0f}"
            )
            battery_surf = self.font.render(battery_text, True, text_color)
            self.screen.blit(battery_surf, (10, start_y))
            current_y = start_y + line_height
            status_text = ""
            status_color = status_color_ok
            if self.landed_safely:
                status_text = "Landed Safely!"
            elif self.crashed:
                status_text = "Crashed!"
                status_color = status_color_bad
            elif self.lost_control:
                status_text = "Lost Control!"
                status_color = status_color_bad
            elif self.out_of_bounds:
                status_text = "Out of Bounds!"
                status_color = status_color_bad
            elif self.Battery_empty:
                status_text = "Battery Empty!"
                status_color = status_color_bad
            elif self.current_step >= self.max_steps:
                status_text = "Time Limit Reached"
                status_color = status_color_bad
            if status_text:
                status_surf = self.font.render(status_text, True, status_color)
                self.screen.blit(status_surf, (10, current_y))
                current_y += line_height
            step_text = f"Step: {self.current_step} / {self.max_steps}"
            step_surf = self.font.render(step_text, True, text_color)
            self.screen.blit(step_surf, (10, current_y))
            current_y += line_height
            raw_pos = self.info.get("raw_pos", (0.0, 0.0))
            raw_vel = self.info.get("raw_vel", (0.0, 0.0))
            angle_rad = self.info.get("raw_angle_rad", 0.0)
            angular_vel = self.info.get("raw_angular_vel", 0.0)
            angle_deg = np.degrees(angle_rad) % 360
            pos_text = f"Pos (X,Y): ({raw_pos[0]:.2f}, {raw_pos[1]:.2f}) m"
            pos_surf = self.font.render(pos_text, True, text_color)
            self.screen.blit(pos_surf, (10, current_y))
            current_y += line_height
            vel_text = f"Vel (Vx,Vy): ({raw_vel[0]:.2f}, {raw_vel[1]:.2f}) m/s"
            vel_surf = self.font.render(vel_text, True, text_color)
            self.screen.blit(vel_surf, (10, current_y))
            current_y += line_height
            angle_text = f"Angle: {angle_deg:.1f}°"
            angle_surf = self.font.render(angle_text, True, text_color)
            self.screen.blit(angle_surf, (10, current_y))
            current_y += line_height
            ang_vel_text = f"Ang Vel: {angular_vel:.2f} rad/s"
            ang_vel_surf = self.font.render(ang_vel_text, True, text_color)
            self.screen.blit(ang_vel_surf, (10, current_y))

        pygame.display.flip()
        if self.clock:
            self.clock.tick(self.metadata["render_fps"])

    # close, change_target_point remain the same
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            print("Pygame closed.")

    def change_target_point(self, x_px, y_px):
        print("Changing target mid-flight not implemented in this version.")
        pass
