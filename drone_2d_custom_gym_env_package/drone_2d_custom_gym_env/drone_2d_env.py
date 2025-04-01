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
        enable_wind: bool = True,               # Master switch for applying wind force. (Values: True, False)
        wind_speed: float = 5.0,                # Base wind speed (m/s) used if enable_wind=True. (Min: 0.0, Rec: 0.0-15.0)
        lander_mass :float =  1.5,               #  measured in kilograms 
        
        
        # --- Reward Shaping ---
        reward_landing: float = 100.0,          # Reward for stable, safe landing. Should be largest positive value. (Min: >0, Rec: 50-1000)
        reward_un_landing: float = 20.0         # Reward for unstable (but safe speed/angle) landing. (Min: 0, Rec: <reward_landing, e.g., 10-50)
    ):
        """
        Initializes the Drone Landing Environment. Args match the parameter list above.
        """
        super().__init__()

        
        # --- Store Initialization Parameters ---
        self.render_sim = render_sim; self.max_steps = max_steps; self.render_path = render_path
        self.render_shade = render_shade; self.shade_distance_m = shade_distance_m
        self.moving_platform = moving_platform; self.platform_speed = platform_speed if moving_platform else 0.0
        self.initial_pos_random_range_m = initial_pos_random_range_m
        self.max_allowed_tilt_angle_rad = max_allowed_tilt_angle_rad
        self.enable_wind = enable_wind
        self.reward_landing = reward_landing; self.reward_un_landing = reward_un_landing

        # === Platform Specific State ===
        self.platform_direction = 1; self.pad_body = None

        # === Environment Physics Parameters ===
        self.gravity_mag = 9.81; self.wind_speed = wind_speed if self.enable_wind else 0.0
        self.wind_force_coefficient = 0.5

        # === Drone Physical Parameters ===
        self.lander_mass = lander_mass; self.lander_width = 1.0; self.lander_height = 0.2
        self.initial_Battery = 100.0; self.max_thrust = 15.0
        self.thrust_noise_std_dev = 0.05; self.Battery_consumption_rate = 0.1

        # === World Dimensions ===
        self.world_width: float = 50.0; self.world_height: float = 50.0; self.ground_height: float = 10.0

        # === Landing Task Parameters ===
        self.landing_pad_width = 5.0; self.landing_pad_height = 0.5
        self.initial_landing_target_x = self.world_width / 2.0; self.landing_target_y = self.ground_height
        self.max_safe_landing_speed = 1.5; self.max_safe_landing_angle = 0.2

        # === Simulation Timing ===
        self.frames_per_second = 50; self.dt = 1.0 / self.frames_per_second

        # === Internal Simulation State ===
        self.current_step = 0; self.landed_safely = False; self.crashed = False
        self.Battery_empty = False; self.out_of_bounds = False; self.lost_control = False
        self.current_left_thrust_applied = 0.0; self.current_right_thrust_applied = 0.0
        self.info = {}; self.current_Battery = self.initial_Battery

        # === Rendering Setup === MODIFIED ===
        self.screen_width_px = screen_width_px * render_zoom_factor  # Use parameter
        self.screen_height_px = screen_height_px * render_zoom_factor    # Use parameter
        # Calculate base pixels per meter to fit world in screen
        base_ppm_x = self.screen_width_px / self.world_width if self.world_width > 0 else 1
        base_ppm_y = self.screen_height_px / self.world_height if self.world_height > 0 else 1
        # Apply zoom factor
        self.pixels_per_meter = min(base_ppm_x, base_ppm_y) * render_zoom_factor # Apply zoom
        print(f"World Size: {self.world_width}m x {self.world_height}m")
        print(f"Screen Size: {self.screen_width_px}px x {self.screen_height_px}px")
        print(f"Render Zoom Factor: {render_zoom_factor}")
        print(f"Effective Pixels Per Meter: {self.pixels_per_meter:.2f}")

        self.screen = None; self.clock = None; self.font = None; self.shade_image = None
        # --- End Rendering Setup Modification ---

        # === Path and Shade Tracking ===
        self.flight_path_px = []; self.path_drone_shade_info = []; self.last_shade_pos = None

        # === Persistent Episode Counters ===
        self.episode_count = 0; self.ep_count_landed = 0; self.ep_count_crashed = 0
        self.ep_count_lost_control = 0; self.ep_count_out_of_bounds = 0
        self.ep_count_battery_empty = 0; self.ep_count_timeout = 0

        if self.render_sim: self.init_pygame() # Must be after pixels_per_meter is set

        # === Action & Observation Spaces === (Definition unchanged)
        min_action = np.array([-1.0, -1.0], dtype=np.float32); max_action = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)
        obs_dim = 11;
        if self.moving_platform: obs_dim = 12
        obs_low_list = [-1,-1,-1,-1,-1,-1,0,-1,-1,-1,0.0]; obs_high_list = [1,1,1,1,1,1,1,1,1,1,1.0]
        if self.moving_platform: obs_low_list.append(-1); obs_high_list.append(1)
        obs_low = np.array(obs_low_list, dtype=np.float32); obs_high = np.array(obs_high_list, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        # print(f"Observation Space Size: {self.observation_space.shape}") # Keep less verbose

        # --- Initialize Physics ---
        self.space = None; self.drone = None; self.landing_pad_shape = None; self.ground_shape = None
        self.seed(); self.reset()

    # seed, _world_to_screen, _screen_to_world, init_pygame, init_pymunk, _add_position_to_path, _add_drone_shade, collision_begin, collision_separate, _apply_forces, _get_observation, _calculate_reward methods remain the same...
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
        # print(f"Drone starting at: ({initial_x:.2f}, {initial_y:.2f}) m") # Less verbose
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
                # print(f"Step {self.current_step}: LANDED SAFELY!") # Less verbose
            else:
                self.crashed = True
                # print(f"Step {self.current_step}: CRASHED on pad! Speed: {velocity.length:.2f}, Angle: {angle:.2f}") # Less verbose
        else:
            self.crashed = True
            # print(f"Step {self.current_step}: CRASHED on ground!") # Less verbose
        return True

    def collision_separate(self, arbiter, space, data):
        self.drone_contacts -= 1
        self.drone_contacts = max(0, self.drone_contacts)

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
        if self.drone:
            self.drone.apply_thrust(applied_left_thrust, applied_right_thrust)
        self.current_left_thrust_applied = applied_left_thrust
        self.current_right_thrust_applied = applied_right_thrust
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
        if self.enable_wind and self.wind_speed > 0:
            wind_force_vec = Vec2d(self.wind_speed, 0).rotated(self.wind_direction)
            wind_force = (
                wind_force_vec * self.wind_force_coefficient * self.lander_width
            )
        if self.drone:
            self.drone.body.apply_force_at_world_point(
                wind_force, self.drone.body.position
            )

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
                reward += self.reward_landing
            else:
                reward += self.reward_un_landing
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

    # MODIFIED _check_termination (to increment counters)
    def _check_termination(self):
        done = False
        terminal_condition_met = False  # Flag to ensure counter increments only once

        # Check drone state only if drone exists
        if self.drone and self.drone.body:
            drone_pos = self.drone.body.position
            angle_rad = self.drone.body.angle

            # Check bounds
            if not (
                0 < drone_pos.x < self.world_width
                and 0 < drone_pos.y < self.world_height
            ):
                if not self.out_of_bounds:
                    print(f"Step {self.current_step}: OUT OF BOUNDS!")
                    self.ep_count_out_of_bounds += 1  # Increment Counter
                    terminal_condition_met = True
                self.out_of_bounds = True
                done = True

            # Check tilt angle (Loss of Control)
            if not done:  # Check only if not already terminated
                angle_norm = (angle_rad + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_norm) > self.max_allowed_tilt_angle_rad:
                    if not self.lost_control:
                        print(f"Step {self.current_step}: LOST CONTROL!")
                        self.ep_count_lost_control += 1  # Increment Counter
                        terminal_condition_met = True
                    self.lost_control = True
                    done = True
        else:
            # If drone doesn't exist, something went wrong, terminate
            print(
                "Warning: Drone object missing in _check_termination, terminating episode."
            )
            if not self.crashed:  # Count as crash if not already flagged
                self.ep_count_crashed += 1
                terminal_condition_met = True
            self.crashed = True
            done = True

        # Check other conditions only if not already done
        if not done and self.Battery_empty and not (self.landed_safely or self.crashed):
            # Battery empty is the primary reason only if not already landed/crashed
            if not terminal_condition_met:
                self.ep_count_battery_empty += 1  # Increment Counter
                terminal_condition_met = True
            done = True

        if not done and (self.landed_safely or self.crashed):
            # Landing/Crash flags are set earlier (collision_begin). Increment counters there?
            # Let's increment here for simplicity, checking flags *after* they might be set.
            if self.landed_safely and not terminal_condition_met:
                self.ep_count_landed += 1  # Increment Counter
                terminal_condition_met = True
            elif (
                self.crashed
                and not terminal_condition_met
                and not self.lost_control
                and not self.out_of_bounds
            ):
                # Only count crash here if not already terminated by LoC or OOB
                self.ep_count_crashed += 1  # Increment Counter
                terminal_condition_met = True
            done = True

        if not done and self.current_step >= self.max_steps:
            if not terminal_condition_met:  # Check if timeout is the primary reason
                print(f"Step {self.current_step}: MAX STEPS REACHED!")
                self.ep_count_timeout += 1  # Increment Counter
                terminal_condition_met = True
            done = True

        return done

    # step remains the same (calls _check_termination)
    def step(self, action):
        if self.drone is None:
            dummy_obs = np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )
            return dummy_obs, 0.0, True, {"error": "Drone not initialized"}
        self._update_platform_position()
        self._apply_forces(action)
        self.space.step(self.dt)

        # --- Calculate distance AND target point for info dict ---
        platform_pos_x = self.initial_landing_target_x
        platform_target_y = self.landing_target_y + self.landing_pad_height
        if self.pad_body:
            platform_pos_x = self.pad_body.position.x

        target_pos_world = (platform_pos_x, platform_target_y)  # Store target point

        distance_to_target_m = -1.0  # Default if drone doesn't exist
        if self.drone and self.drone.body:
            drone_pos = self.drone.body.position
            world_vector_to_target = Vec2d(
                target_pos_world[0] - drone_pos.x, target_pos_world[1] - drone_pos.y
            )
            distance_to_target_m = world_vector_to_target.length
        # --- End calculation for info ---

        current_pos = self.drone.body.position
        self._add_position_to_path()
        if self.render_shade and self.last_shade_pos is not None:
            if current_pos.get_distance(self.last_shade_pos) > self.shade_distance_m:
                self._add_drone_shade()

        observation = self._get_observation()
        done = self._check_termination()
        reward = self._calculate_reward(observation)
        self.current_step += 1
        drone_body = self.drone.body if self.drone else None

        # Ensure drone_body exists before accessing attributes for info
        raw_pos_tuple = tuple(drone_body.position) if drone_body else (0.0, 0.0)
        raw_vel_tuple = tuple(drone_body.velocity) if drone_body else (0.0, 0.0)
        raw_angle_val = drone_body.angle if drone_body else 0.0
        raw_ang_vel_val = drone_body.angular_velocity if drone_body else 0.0

        # Update info dictionary
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
            "platform_pos_x": platform_pos_x,  # Use calculated platform pos
            "platform_vel_x": self.pad_body.velocity.x if self.pad_body else 0.0,
            "distance_to_target": distance_to_target_m,
            "target_pos_world": target_pos_world,  # ADDED target coordinates
        }
        return observation, reward, done, self.info

    # MODIFIED reset
    def reset(self):
        # --- Increment Episode Counter ---
        self.episode_count += 1
        # --- End Increment ---

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

        # Reset PER-EPISODE state vars (DO NOT RESET COUNTERS)
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

        # Reset path/shade
        self.flight_path_px = []
        self.path_drone_shade_info = []
        self.last_shade_pos = None

        # Randomize wind/platform direction
        self.wind_direction = self.np_random.uniform(0, 2 * np.pi)
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
            print("Error: Drone not created during reset.")

        # --- Print Reset Info (less frequent maybe) ---
        if (
            self.episode_count % 10 == 1 or self.episode_count <= 1
        ):  # Print every 10 eps or first ep
            print(f"\n--- Resetting Episode {self.episode_count} ---")
            print(
                f"Wind Enabled: {self.enable_wind}, Wind Dir: {np.degrees(self.wind_direction):.1f} deg. Moving Platform: {self.moving_platform}"
            )
        # --- End Print ---

        # Return initial observation
        if self.drone:
            return self._get_observation()
        else:
            return np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )

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

        # ... (Keep Drawing Spawn Zone, Shades, Paths, Ground, Pad, Drone, Vectors, Wind) ...
        # --- Draw Initial Spawn Zone ---
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
        # --- Draw Shades FIRST ---
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
        # --- Draw Paths ---
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
        # --- Draw Motor Force Vectors ---
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
        # --- Draw Wind Indicator and Label ---
        if self.enable_wind and self.wind_speed > 0:
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
        if self.font and self.enable_wind and self.wind_speed > 0:
            wind_dir_degrees = np.degrees(self.wind_direction) % 360
            wind_label_text = (
                f"Wind: {self.wind_speed:.1f} m/s @ {wind_dir_degrees:.0f}°"
            )
            wind_label_surf = self.font.render(wind_label_text, True, wind_label_color)
            label_pos = (
                wind_indicator_base_pos[0] - wind_label_surf.get_width() - 10,
                wind_indicator_base_pos[1] + 5,
            )
            self.screen.blit(wind_label_surf, label_pos)

        # --- Draw Sensor Vector Line --- ADDED ---
        if self.drone and self.info:  # Check if drone and info exist
            drone_pos_world = self.info.get("raw_pos")
            target_pos_world = self.info.get("target_pos_world")

            if drone_pos_world and target_pos_world:  # Check if positions are valid
                try:
                    start_screen = self._world_to_screen(drone_pos_world)
                    end_screen = self._world_to_screen(target_pos_world)
                    sensor_line_color = (0, 180, 180)  # Cyan color
                    sensor_line_thickness = 1
                    # Draw the line
                    pygame.draw.aaline(
                        self.screen, sensor_line_color, start_screen, end_screen
                    )  # Use aaline for anti-aliasing
                    # Optional: Draw small circles at ends
                    # pygame.draw.circle(self.screen, sensor_line_color, start_screen, 3)
                    # pygame.draw.circle(self.screen, (255,0,0), end_screen, 3) # Red target circle
                except Exception as e:
                    print(f"Error drawing sensor vector: {e}")
        # --- End Sensor Vector Line ---

        # --- Draw Info Text ---
        if self.font:
            start_y = 10
            line_height = 20
            text_color = (0, 0, 0)
            status_color_ok = (0, 150, 0)
            status_color_bad = (200, 0, 0)
            current_y = start_y
            # Current Step Info
            battery_text = (
                f"Battery: {self.current_Battery:.1f}/{self.initial_Battery:.0f}"
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
            step_text = f"Step: {self.current_step}/{self.max_steps}"
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
            # Cumulative Stats Text
            current_y += line_height  # Add gap
            ep_text = f"Total Episodes: {self.episode_count}"
            ep_surf = self.font.render(ep_text, True, text_color)
            self.screen.blit(ep_surf, (10, current_y))
            current_y += line_height
            total_finished = max(1, self.episode_count - 1)
            success_rate = (
                self.ep_count_landed / total_finished if total_finished > 0 else 0.0
            )
            stats_text1 = f"Landed:{self.ep_count_landed}({success_rate:.1%})|Crashed:{self.ep_count_crashed}"
            stats_surf1 = self.font.render(stats_text1, True, text_color)
            self.screen.blit(stats_surf1, (10, current_y))
            current_y += line_height
            stats_text2 = f"LoC:{self.ep_count_lost_control}|OoB:{self.ep_count_out_of_bounds}|Bat:{self.ep_count_battery_empty}|Timeout:{self.ep_count_timeout}"
            stats_surf2 = self.font.render(stats_text2, True, text_color)
            self.screen.blit(stats_surf2, (10, current_y))
            # current_y+=line_height

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