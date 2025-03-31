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
        # --- Rendering & Simulation Control ---
        render_sim: bool = False,               # Master switch for enabling Pygame visualization. Set False for training. (Values: True, False)
        max_steps: int = 1000,                  # Max simulation steps per episode before timeout. (Min: ~100, Rec: 500-2000)
        render_path: bool = True,               # Draw drone's trajectory history if render_sim=True. (Values: True, False)
        render_shade: bool = True,              # Draw drone "shades" along path if render_sim=True. Requires img/shade.png. (Values: True, False)
        shade_distance_m: float = 2.0,          # Min distance (m) drone travels before dropping a new shade. (Min: >0, Rec: 0.5-5.0)
        render_zoom_factor: float = 1.0,        # Visual zoom level (1.0 = fit world). Affects visuals only. (Min: >0, Rec: 0.5-3.0)
        screen_width_px: int = 800,             # Pygame window width in pixels. (Min: ~400, Rec: 600-1600)
        screen_height_px: int = 800 ,            # Pygame window height in pixels. (Min: ~400, Rec: 600-1200)
        

        # --- Platform Configuration ---
        moving_platform: bool = False,          # If True, platform moves horizontally. (Values: True, False)
        platform_speed: float = 1.5,            # Horizontal speed (m/s) of platform if moving_platform=True. (Min: 0.0, Rec: 0.5-5.0)

        # --- Environment Dynamics & Constraints ---
        initial_pos_random_range_m: float = 5.0,# Half-range (m) for randomizing start position around center. 0.0 for fixed start. (Min: 0.0, Rec: 0.0-15.0)
        max_allowed_tilt_angle_rad: float = np.pi / 2.0, # Max absolute tilt (radians) before "Lost Control" termination. (Min: >0, Rec: pi/3-pi/2 [60-90 deg])
        center_on_drone: bool = True,           # If True and rendering+zoomed, keeps drone centered in view.
        lander_mass :float =  1.5,               #  measured in kilograms 
        
        # --- WIND Configurations ---
        # --- WIND Basics ---        
        enable_wind: bool = True,               # Master switch for applying wind force. (Values: True, False)
        wind_speed: float = 5.0,                # Base wind speed (m/s) used if enable_wind=True. (Min: 0.0, Rec: 0.0-15.0)
        
         # --- Wind Advance ---
        enable_dynamic_wind: bool = False,      # If True, wind speed/direction can change during the episode (overrides initial wind_speed after first change).
        dynamic_wind_change_prob: float = 0.005,  # Probability per step of triggering a wind change (if dynamic wind enabled). Rec: 0.001-0.02
        dynamic_wind_speed_range: tuple = (0.0, 8.0),  # Min/Max wind speed (m/s) when a dynamic change occurs.
        dynamic_wind_change_both: bool = False,  # If True, change speed & direction together; False, change one randomly.
        
        # --- Reward Shaping ---
        reward_landing: float = 100.0,          # Reward for stable, safe landing. Should be largest positive value. (Min: >0, Rec: 50-1000)
        reward_un_landing: float = 20.0 ,        # Reward for unstable (but safe speed/angle) landing. (Min: 0, Rec: <reward_landing, e.g., 10-50)
    ):
        """
        Initializes the Drone Landing Environment. Args match the parameter list above.
        """
        super().__init__()

        
        # --- Store Initialization Parameters ---
        # Rendering & Sim Control
        self.render_sim = render_sim
        self.max_steps = max_steps
        self.render_path = render_path
        self.render_shade = render_shade
        self.shade_distance_m = shade_distance_m
        self.screen_width_px = screen_width_px * render_zoom_factor
        self.screen_height_px = screen_height_px * render_zoom_factor
        self.render_zoom_factor = render_zoom_factor
        self.center_on_drone = center_on_drone
        # Platform
        self.moving_platform = moving_platform
        self.platform_speed = platform_speed if moving_platform else 0.0
        # Dynamics & Constraints
        self.initial_pos_random_range_m = initial_pos_random_range_m
        self.max_allowed_tilt_angle_rad = max_allowed_tilt_angle_rad
        self.world_width = 50.0
        self.world_height = 50.0
        # Wind
        self.enable_wind = enable_wind
        self.base_wind_speed = wind_speed  # Store the initial speed parameter
        self.enable_dynamic_wind = enable_dynamic_wind
        self.dynamic_wind_change_prob = dynamic_wind_change_prob
        self.dynamic_wind_speed_range = dynamic_wind_speed_range
        self.dynamic_wind_change_both = dynamic_wind_change_both
        # Reward
        self.reward_landing = reward_landing
        self.reward_un_landing = reward_un_landing

        # === Platform Specific State ===
        self.platform_direction = 1
        self.pad_body = None

        # === Environment Physics Parameters ===
        self.gravity_mag = 9.81
        # Current wind state (initialized in reset)
        self.current_wind_speed: float = 0.0
        self.current_wind_direction: float = 0.0  # Will be set in seed/reset
        self.wind_force_coefficient = 0.5

        # === Drone Physical Parameters ===
        self.lander_mass = lander_mass
        self.lander_width = 1.0 
        self.lander_height = 0.2 
        self.initial_Battery = 100.0
        self.max_thrust = 15.0
        self.thrust_noise_std_dev = 0.05
        self.Battery_consumption_rate = 0.1

        # === World Geometry ===
        self.ground_height = 10.0

        # === Landing Task Parameters ===
        self.landing_pad_width = 5.0
        self.landing_pad_height = 0.5
        self.initial_landing_target_x = self.world_width / 2.0
        self.landing_target_y = self.ground_height
        self.max_safe_landing_speed = 1.5
        self.max_safe_landing_angle = 0.2

        # === Simulation Timing ===
        self.frames_per_second = 50
        self.dt = 1.0 / self.frames_per_second

        # === Internal Simulation State ===
        self.current_step = 0
        self.landed_safely = False
        self.crashed = False
        self.Battery_empty = False
        self.out_of_bounds = False
        self.lost_control = False
        self.current_left_thrust_applied = 0.0
        self.current_right_thrust_applied = 0.0
        self.info = {}
        self.current_Battery = self.initial_Battery

        # === Rendering Setup ===
        base_ppm_x = (
            self.screen_width_px / self.world_width if self.world_width > 1e-6 else 1
        )
        base_ppm_y = (
            self.screen_height_px / self.world_height if self.world_height > 1e-6 else 1
        )
        self.pixels_per_meter = min(base_ppm_x, base_ppm_y) * self.render_zoom_factor
        self.screen = None
        self.clock = None
        self.font = None
        self.shade_image = None

        # === Path and Shade Tracking ===
        self.flight_path_px = []
        self.path_drone_shade_info = []
        self.last_shade_pos = None

        # === Persistent Episode Counters ===
        self.episode_count = 0
        self.ep_count_landed = 0
        self.ep_count_crashed = 0
        self.ep_count_lost_control = 0
        self.ep_count_out_of_bounds = 0
        self.ep_count_battery_empty = 0
        self.ep_count_timeout = 0

        if self.render_sim:
            self.init_pygame()

        # === Action & Observation Spaces ===
        min_action = np.array([-1.0, -1.0], dtype=np.float32)
        max_action = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=min_action, high=max_action, dtype=np.float32
        )
        obs_dim = 11
        if self.moving_platform:
            obs_dim = 12
        obs_low_list = [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0.0]
        obs_high_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0]
        if self.moving_platform:
            obs_low_list.append(-1)
            obs_high_list.append(1)
        obs_low = np.array(obs_low_list, dtype=np.float32)
        obs_high = np.array(obs_high_list, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        # print(f"Observation Space Size: {self.observation_space.shape}")

        # --- Initialize Physics ---
        self.space = None
        self.drone = None
        self.landing_pad_shape = None
        self.ground_shape = None
        self.seed()
        self.reset()

    # --- Methods ---
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # Initialize wind direction here AFTER np_random is created
        self.current_wind_direction = self.np_random.uniform(0, 2 * np.pi)
        if self.moving_platform:
            self.platform_direction = self.np_random.choice([-1, 1])
        return [seed]

    def _world_to_screen(self, world_pos):
        wx, wy = world_pos
        base_sx = wx * self.pixels_per_meter
        base_sy = self.screen_height_px - (wy * self.pixels_per_meter)
        offset_x = 0.0
        offset_y = 0.0
        if self.render_sim and self.center_on_drone and self.drone and self.drone.body:
            drone_w_x, drone_w_y = self.drone.body.position
            drone_base_sx = drone_w_x * self.pixels_per_meter
            drone_base_sy = self.screen_height_px - (drone_w_y * self.pixels_per_meter)
            screen_center_x = self.screen_width_px / 2.0
            screen_center_y = self.screen_height_px / 2.0
            offset_x = screen_center_x - drone_base_sx
            offset_y = screen_center_y - drone_base_sy
        final_sx = base_sx + offset_x
        final_sy = base_sy + offset_y
        return int(final_sx), int(final_sy)

    def _screen_to_world(self, screen_pos):
        sx, sy = screen_pos
        wx = sx / self.pixels_per_meter
        wy = (self.screen_height_px - sy) / self.pixels_per_meter
        if self.center_on_drone:
            print(
                "Warning: _screen_to_world conversion is inaccurate when center_on_drone=True"
            )
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
            original_shade_image = None
            try:
                script_dir = os.path.dirname(__file__)
                img_path = os.path.join(script_dir, "img", "shade.png")
                original_shade_image = pygame.image.load(img_path).convert_alpha()
            except Exception as e:
                print(f"Warning: Error loading shade image {img_path}: {e}.")
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
                        print("Warning: Calculated zero size for shade image.")
                        self.shade_image = None
                except Exception as e:
                    print(f"Warning: Error scaling shade image: {e}.")
                    self.shade_image = None
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
        try:
            self.drone = Drone(
                initial_x,
                initial_y,
                initial_angle,
                self.lander_height,
                self.lander_width,
                self.lander_mass,
                self.space,
            )
        except Exception as e:
            print(f"!!! CRITICAL ERROR creating Drone object: {e} !!!")
            self.drone = None
        if self.drone:
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
        if (
            not self.render_sim
            or not self.render_path
            or not self.drone
            or not self.drone.body
        ):
            return
        pos_screen = self._world_to_screen(self.drone.body.position)
        self.flight_path_px.append(pos_screen)

    def _add_drone_shade(self):
        if (
            not self.render_sim
            or not self.render_shade
            or not self.drone
            or not self.drone.body
        ):
            return
        pos_world = self.drone.body.position
        angle_rad = self.drone.body.angle
        self.path_drone_shade_info.append([pos_world.x, pos_world.y, angle_rad])
        self.last_shade_pos = pos_world

    def collision_begin(self, arbiter, space, data):
        if not self.drone or not self.drone.body:
            return True
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
            else:
                self.crashed = True
        else:
            self.crashed = True
        return True

    def collision_separate(self, arbiter, space, data):
        self.drone_contacts -= 1
        self.drone_contacts = max(0, self.drone_contacts)

    # --- MODIFIED _apply_forces ---
    def _apply_forces(self, action):
        if not self.drone or not self.drone.body:
            return

        # Calculate and Apply Thrust
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
        self.drone.apply_thrust(applied_left_thrust, applied_right_thrust)
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

        # Apply Wind (uses current dynamic wind state)
        if self.enable_wind and self.current_wind_speed > 0:  # Use current_wind_speed
            wind_force_vec = Vec2d(self.current_wind_speed, 0).rotated(
                self.current_wind_direction
            )
            wind_force = (
                wind_force_vec * self.wind_force_coefficient * self.lander_width
            )
            self.drone.body.apply_force_at_world_point(
                wind_force, self.drone.body.position
            )

    # --- End Modification ---

    def _get_observation(self):
        if not self.drone or not self.drone.body:
            return np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )
        drone_body = self.drone.body
        pos = drone_body.position
        vel = drone_body.velocity
        angle = drone_body.angle
        angular_vel = drone_body.angular_velocity
        platform_pos_x = self.initial_landing_target_x
        platform_target_y = self.landing_target_y + self.landing_pad_height
        platform_vel_x = 0.0
        if self.pad_body:
            platform_pos_x = self.pad_body.position.x
        if self.pad_body and self.moving_platform:
            platform_vel_x = self.pad_body.velocity.x
        target_pos_world = (platform_pos_x, platform_target_y)
        world_vector_to_target = Vec2d(
            target_pos_world[0] - pos.x, target_pos_world[1] - pos.y
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
        norm_angle_obs = (angle + np.pi) % (2 * np.pi) - np.pi
        norm_angle_obs = np.clip(norm_angle_obs / np.pi, -1.0, 1.0)
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
            norm_angle_obs,
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
        if obs.shape[0] != self.observation_space.shape[0]:
            print(
                f"!!! CRITICAL WARNING: Final observation shape mismatch. Got {obs.shape}, expected {self.observation_space.shape} !!!"
            )
            correct_obs = np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )
            copy_len = min(len(obs), len(correct_obs))
            correct_obs[:copy_len] = obs[:copy_len]
            obs = correct_obs
            self.crashed = True
        if np.isnan(obs).any() or np.isinf(obs).any():
            print("!!! WARNING: NaN or Inf detected in observation !!!")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            self.crashed = True
        return obs

    def _calculate_reward(self, observation):
        # --- Determine expected length ---
        expected_len = 11  # Base + Geometric Sensor
        if self.moving_platform:
            expected_len = 12

        if len(observation) != expected_len:
            print(
                f"!!! WARNING: Obs length mismatch in reward. Expected {expected_len}, got {len(observation)} !!!"
            )
            return 0.0  # Return 0 reward if observation is malformed

        # --- Unpack observation ---
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

        # --- Calculate Dense Rewards ---
        reward = 0.0
        # Proximity Reward (using normalized distance directly)
        # Higher reward when norm_distance is closer to 0.0
        reward += 2.0 * np.exp(-5.0 * norm_distance)  # Tune the -5.0 factor

        # Penalties for instability/velocity
        reward -= (norm_vel_x**2 + norm_vel_y**2) * 0.1  # Velocity penalty
        reward -= abs(norm_angle) * 0.2  # Angle penalty
        reward -= abs(norm_angular_vel) * 0.1  # Angular velocity penalty

        # --- Terminal Rewards/Penalties --- CORRECTED STRUCTURE ---
        if self.landed_safely:
            # Check for stability only if landed safely
            drone_body = self.drone.body if self.drone else None
            if (
                drone_body
                and drone_body.velocity.length < 0.1
                and abs(drone_body.angular_velocity) < 0.1
            ):
                reward += self.reward_landing  # Stable landing reward
            else:
                reward += self.reward_un_landing  # Unstable landing reward
        # Only apply penalties if NOT landed safely
        elif self.crashed:
            reward -= 50.0
        elif self.lost_control:
            reward -= 50.0
        elif (
            self.Battery_empty
        ):  # Check Battery empty specifically if not crashed/LoC/OOB
            reward -= 50.0
        elif self.out_of_bounds:
            reward -= 50.0
        elif self.current_step >= self.max_steps:  # Check timeout last
            reward -= 10.0
        # --- End Correction ---

        return reward

    def _check_termination(self):
        done = False
        terminal_condition_met = False
        drone_pos = None
        angle_rad = 0.0
        if self.drone and self.drone.body:
            drone_pos = self.drone.body.position
            angle_rad = self.drone.body.angle
        else:
            print("Warning: Drone object missing in _check_termination.")
            done = True
            self.crashed = True
            self.ep_count_crashed += 1
            terminal_condition_met = True
        if not done and drone_pos:
            if not (
                0 < drone_pos.x < self.world_width
                and 0 < drone_pos.y < self.world_height
            ):
                if not self.out_of_bounds:
                    self.ep_count_out_of_bounds += 1
                    terminal_condition_met = True
                    print(f"Step {self.current_step}: OUT OF BOUNDS!")
                self.out_of_bounds = True
                done = True
            if not done:
                angle_norm = (angle_rad + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_norm) > self.max_allowed_tilt_angle_rad:
                    if not self.lost_control:
                        self.ep_count_lost_control += 1
                        terminal_condition_met = True
                        print(f"Step {self.current_step}: LOST CONTROL!")
                    self.lost_control = True
                    done = True
        if not done and self.Battery_empty and not (self.landed_safely or self.crashed):
            if not terminal_condition_met:
                self.ep_count_battery_empty += 1
                terminal_condition_met = True
            done = True
        if not done and (self.landed_safely or self.crashed):
            if self.landed_safely and not terminal_condition_met:
                self.ep_count_landed += 1
                terminal_condition_met = True
            elif (
                self.crashed
                and not terminal_condition_met
                and not self.lost_control
                and not self.out_of_bounds
            ):
                self.ep_count_crashed += 1
                terminal_condition_met = True
            done = True
        if not done and self.current_step >= self.max_steps:
            if not terminal_condition_met:
                self.ep_count_timeout += 1
                terminal_condition_met = True
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

    # --- ADDED: Dynamic Wind Update Logic ---
    def _update_dynamic_wind(self):
        """Checks probability and potentially updates wind speed and/or direction."""
        if not self.enable_dynamic_wind or not self.enable_wind:
            return  # Exit if dynamic wind or global wind is disabled

        # Check if a change should occur this step
        if self.np_random.random() < self.dynamic_wind_change_prob:
            changed_speed = False
            changed_dir = False

            # Determine what to change
            change_speed = False
            change_direction = False
            if self.dynamic_wind_change_both:
                change_speed = True
                change_direction = True
            else:
                # Randomly choose to change speed or direction
                if self.np_random.choice([0, 1]) == 0:
                    change_speed = True
                else:
                    change_direction = True

            # Apply changes
            if change_speed:
                min_w, max_w = self.dynamic_wind_speed_range
                self.current_wind_speed = self.np_random.uniform(min_w, max_w)
                changed_speed = True
            if change_direction:
                self.current_wind_direction = self.np_random.uniform(0, 2 * np.pi)
                changed_dir = True

            # Optional: Print info about the change
            if changed_speed or changed_dir:
                print(
                    f"Step {self.current_step}: Wind Changed! Speed: {self.current_wind_speed:.2f} m/s, Dir: {np.degrees(self.current_wind_direction):.1f} deg"
                )

    # --- MODIFIED step ---
    def step(self, action):
        if not self.drone or not self.drone.body:
            dummy_obs = np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )
            return dummy_obs, 0.0, True, {"error": "Drone not initialized in step"}

        # --- Update Dynamic Wind FIRST --- ADDED ---
        self._update_dynamic_wind()
        # --- End Update ---

        self._update_platform_position()
        self._apply_forces(
            action
        )  # Uses self.current_wind_speed / self.current_wind_direction
        self.space.step(self.dt)

        # Calculate distance for info dict
        platform_pos_x = self.initial_landing_target_x
        platform_target_y = self.landing_target_y + self.landing_pad_height
        if self.pad_body:
            platform_pos_x = self.pad_body.position.x
        target_pos_world = (platform_pos_x, platform_target_y)
        distance_to_target_m = -1.0
        drone_pos = self.drone.body.position
        distance_to_target_m = (Vec2d(*target_pos_world) - drone_pos).length

        # Path and Shade updates
        self._add_position_to_path()
        if self.render_shade and self.last_shade_pos is not None:
            if drone_pos.get_distance(self.last_shade_pos) > self.shade_distance_m:
                self._add_drone_shade()

        observation = self._get_observation()
        done = self._check_termination()
        reward = self._calculate_reward(observation)
        self.current_step += 1
        drone_body = self.drone.body
        raw_pos_tuple = tuple(drone_body.position)
        raw_vel_tuple = tuple(drone_body.velocity)
        raw_angle_val = drone_body.angle
        raw_ang_vel_val = drone_body.angular_velocity
        self.info = {
            "Battery": self.current_Battery,
            "landed": self.landed_safely,
            "crashed": self.crashed,
            "out_of_bounds": self.out_of_bounds,
            "Battery_empty": self.Battery_empty,
            "lost_control": self.lost_control,
            "steps": self.current_step,
            "raw_pos": raw_pos_tuple,
            "raw_vel": raw_vel_tuple,
            "raw_angle_rad": raw_angle_val,
            "raw_angular_vel": raw_ang_vel_val,
            "platform_pos_x": platform_pos_x,
            "platform_vel_x": self.pad_body.velocity.x if self.pad_body else 0.0,
            "distance_to_target": distance_to_target_m,
            "target_pos_world": target_pos_world,
        }
        return observation, reward, done, self.info

    # MODIFIED reset
    def reset(self):
        self.episode_count += 1
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
        # Reset PER-EPISODE state vars
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
        # Reset path/shade/rays
        self.flight_path_px = []
        self.path_drone_shade_info = []
        self.last_shade_pos = None
        self.last_ray_starts_world = []
        self.last_ray_ends_world = []
        self.last_ray_hit_platform = []

        # --- Initialize Wind State --- MODIFIED ---
        self.current_wind_speed = self.base_wind_speed if self.enable_wind else 0.0
        self.current_wind_direction = self.np_random.uniform(
            0, 2 * np.pi
        )  # Always set initial direction
        # --- End Wind Init ---

        # Init platform direction
        if self.moving_platform:
            self.platform_direction = self.np_random.choice([-1, 1])
        else:
            self.platform_direction = 1

        # Re-init pymunk
        self.init_pymunk()
        # Add initial path/shade
        if self.drone:
            self._add_position_to_path()
            self._add_drone_shade()
        else:
            print("Error: Drone object is None after init_pymunk in reset.")

        # Print Reset Info
        if self.episode_count % 10 == 1 or self.episode_count <= 1:
            print(f"\n--- Resetting Episode {self.episode_count} ---")
            print(
                f"Wind Enabled: {self.enable_wind}, Initial Speed: {self.current_wind_speed:.1f} m/s, Dynamic Wind: {self.enable_dynamic_wind}"
            )
            print(f"Moving Platform: {self.moving_platform}")

        # Return initial observation
        if self.drone:
            return self._get_observation()
        else:
            return np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )

    # MODIFIED render (uses current_wind_speed/direction)
    def render(self, mode="human"):
        if not self.render_sim or self.screen is None:
            return
        drone_exists = (
            self.drone is not None
            and hasattr(self.drone, "body")
            and self.drone.body is not None
        )
        if not drone_exists:
            print(
                "Warning: Trying to render but drone/body is not initialized properly."
            )
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

        # --- Draw Background Elements ---
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
        if self.render_path and len(self.flight_path_px) > 1:
            pygame.draw.aalines(self.screen, (16, 19, 97), False, self.flight_path_px)
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

        # --- Draw Drone Shapes ---
        drone_body = self.drone.body
        num_shapes_drawn = 0
        for shape in self.drone.shapes:
            body = shape.body
            if isinstance(shape, pymunk.Poly):
                try:
                    verts_world = [body.local_to_world(v) for v in shape.get_vertices()]
                    verts_screen = [self._world_to_screen(v) for v in verts_world]
                    color = shape.color if hasattr(shape, "color") else (100, 100, 200)
                    if len(verts_screen) > 2:
                        pygame.draw.polygon(self.screen, color, verts_screen)
                        pygame.draw.polygon(self.screen, (0, 0, 0), verts_screen, 1)
                        num_shapes_drawn += 1
                except Exception as e:
                    print(f"  ERROR drawing drone shape: {e}")

        # --- Calculate Visual Offset Point ---
        vis_line_local_offset = Vec2d(0, -self.lander_height * 0.6)
        vis_line_start_world = drone_body.local_to_world(vis_line_local_offset)
        vis_line_start_screen = self._world_to_screen(vis_line_start_world)

        # --- Draw Raycasting Lines --- (No changes needed here)
        if self.enable_raycasting and self.last_ray_starts_world:
            color_hit_platform = (0, 255, 0, 150)
            color_miss = (200, 200, 200, 100)
            ray_thickness = 1
            for i in range(len(self.last_ray_starts_world)):
                try:
                    start_px = vis_line_start_screen
                    end_pt_world = self.last_ray_ends_world[i]
                    hit_platform = self.last_ray_hit_platform[i]
                    end_px = self._world_to_screen(end_pt_world)
                    line_color = color_hit_platform if hit_platform else color_miss
                    line_surf = self.screen.convert_alpha()
                    line_surf.fill((0, 0, 0, 0))
                    pygame.draw.line(
                        line_surf, line_color, start_px, end_px, ray_thickness
                    )
                    self.screen.blit(line_surf, (0, 0))
                except Exception as e:
                    print(f"Error drawing ray {i}: {e}")

        # --- Draw Sensor Vector Line --- (No changes needed here)
        target_pos_world = None
        if self.info:
            target_pos_world = self.info.get("target_pos_world")
        if vis_line_start_world and target_pos_world:
            try:
                start_screen = vis_line_start_screen
                end_screen = self._world_to_screen(target_pos_world)
                sensor_line_color = (0, 180, 180)
                pygame.draw.aaline(
                    self.screen, sensor_line_color, start_screen, end_screen
                )
            except Exception as e:
                print(f"Error drawing sensor vector: {e}")

        # --- Draw Motor Force Vectors --- (No changes needed here)
        if self.drone:
            max_thrust_render_length_px = 40
            grey_color = (170, 170, 170)
            red_color = (255, 50, 50)
            line_width = 3
            max_thrust_render_length_m = (
                max_thrust_render_length_px / self.pixels_per_meter
            )
            max_thrust_val = self.max_thrust if self.max_thrust > 1e-6 else 1.0
            try:
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
            try:
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

        # --- Draw Wind Indicator and Label --- MODIFIED (Uses current_wind_speed/direction)
        if self.enable_wind and self.current_wind_speed > 0:  # Check current speed
            wind_indicator_base_pos = (self.screen_width_px - 70, 50)
            wind_color = (50, 50, 200)
            wind_label_color = (0, 0, 0)
            # Scale arrow length based on current speed relative to max possible dynamic speed (or base speed if not dynamic)
            max_possible_wind = (
                self.dynamic_wind_speed_range[1]
                if self.enable_dynamic_wind
                else self.base_wind_speed
            )
            if max_possible_wind < 1e-6:
                max_possible_wind = 1.0  # Avoid division by zero
            arrow_scale = self.current_wind_speed / max_possible_wind
            arrow_length_px = min(arrow_scale * 60, 60)  # Max length 60px, scales down

            wind_vec_screen = Vec2d(arrow_length_px, 0).rotated(
                -self.current_wind_direction
            )  # Use current direction
            wind_end_px = (
                wind_indicator_base_pos[0] + wind_vec_screen.x,
                wind_indicator_base_pos[1] + wind_vec_screen.y,
            )
            pygame.draw.line(
                self.screen, wind_color, wind_indicator_base_pos, wind_end_px, 3
            )
            pygame.draw.circle(self.screen, wind_color, wind_indicator_base_pos, 5)
        if (
            self.font and self.enable_wind
        ):  # Show label even if speed is 0 if wind enabled
            wind_dir_degrees = (
                np.degrees(self.current_wind_direction) % 360
            )  # Use current direction
            wind_label_text = f"Wind: {self.current_wind_speed:.1f} m/s @ {wind_dir_degrees:.0f}°"  # Use current speed
            wind_label_surf = self.font.render(wind_label_text, True, wind_label_color)
            label_pos = (
                wind_indicator_base_pos[0] - wind_label_surf.get_width() - 10,
                wind_indicator_base_pos[1] + 5,
            )
            self.screen.blit(wind_label_surf, label_pos)
        # --- End Wind Indicator Modification ---

        # --- Draw Info Text --- (No changes needed here)
        if self.font:
            start_y = 10
            line_height = 20
            text_color = (0, 0, 0)
            status_color_ok = (0, 150, 0)
            status_color_bad = (200, 0, 0)
            current_y = start_y
            battery_text = (
                f"Battery:{self.current_Battery:.1f}/{self.initial_Battery:.0f}"
            )
            battery_surf = self.font.render(battery_text, True, text_color)
            self.screen.blit(battery_surf, (10, current_y))
            current_y += line_height
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
            elif self.current_step >= self.max_steps and not (
                self.landed_safely
                or self.crashed
                or self.lost_control
                or self.out_of_bounds
                or self.Battery_empty
            ):
                status_text = "Time Limit Reached"
                status_color = status_color_bad
            if status_text:
                status_surf = self.font.render(status_text, True, status_color)
                self.screen.blit(status_surf, (10, current_y))
                current_y += line_height
            step_text = f"Step:{self.current_step}/{self.max_steps}"
            step_surf = self.font.render(step_text, True, text_color)
            self.screen.blit(step_surf, (10, current_y))
            current_y += line_height
            raw_pos = self.info.get("raw_pos", (0.0, 0.0))
            raw_vel = self.info.get("raw_vel", (0.0, 0.0))
            angle_rad = self.info.get("raw_angle_rad", 0.0)
            angular_vel = self.info.get("raw_angular_vel", 0.0)
            angle_deg = np.degrees(angle_rad) % 360
            pos_text = f"Pos(X,Y):({raw_pos[0]:.2f},{raw_pos[1]:.2f})m"
            pos_surf = self.font.render(pos_text, True, text_color)
            self.screen.blit(pos_surf, (10, current_y))
            current_y += line_height
            vel_text = f"Vel(Vx,Vy):({raw_vel[0]:.2f},{raw_vel[1]:.2f})m/s"
            vel_surf = self.font.render(vel_text, True, text_color)
            self.screen.blit(vel_surf, (10, current_y))
            current_y += line_height
            angle_text = f"Angle:{angle_deg:.1f}°"
            angle_surf = self.font.render(angle_text, True, text_color)
            self.screen.blit(angle_surf, (10, current_y))
            current_y += line_height
            ang_vel_text = f"AngVel:{angular_vel:.2f}rad/s"
            ang_vel_surf = self.font.render(ang_vel_text, True, text_color)
            self.screen.blit(ang_vel_surf, (10, current_y))
            current_y += line_height
            dist_m = self.info.get("distance_to_target", -1.0)
            dist_text = (
                f"Dist Target:{dist_m:.2f}m" if dist_m >= 0 else "Dist Target:N/A"
            )
            dist_surf = self.font.render(dist_text, True, text_color)
            self.screen.blit(dist_surf, (10, current_y))
            current_y += line_height
            current_y += line_height  # Add gap
            ep_text = f"Total Eps:{self.episode_count}"
            ep_surf = self.font.render(ep_text, True, text_color)
            self.screen.blit(ep_surf, (10, current_y))
            current_y += line_height
            total_finished = max(1, self.episode_count - 1)
            success_rate = (
                self.ep_count_landed / total_finished if total_finished > 0 else 0.0
            )
            stats_text1 = f"Landed:{self.ep_count_landed}({success_rate:.1%})|Crash:{self.ep_count_crashed}"
            stats_surf1 = self.font.render(stats_text1, True, text_color)
            self.screen.blit(stats_surf1, (10, current_y))
            current_y += line_height
            stats_text2 = f"LoC:{self.ep_count_lost_control}|OoB:{self.ep_count_out_of_bounds}|Bat:{self.ep_count_battery_empty}|Timeout:{self.ep_count_timeout}"
            stats_surf2 = self.font.render(stats_text2, True, text_color)
            self.screen.blit(stats_surf2, (10, current_y))

        pygame.display.flip()
        if self.clock:
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            print("Pygame closed.")

    def change_target_point(self, x_px, y_px):
        print("Changing target mid-flight not implemented in this version.")
        pass
