import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import pygame

class Drone():
    # Changed parameters: Use total mass and dimensions in meters
    def __init__(self, x, y, angle, height_m, width_m, total_mass_kg, space):
        # Distribute total mass (example: 20% frame, 40% each motor)
        mass_f = total_mass_kg * 0.2
        mass_l = total_mass_kg * 0.4
        mass_r = total_mass_kg * 0.4

        # Dimensions in meters
        self.width_m = width_m
        self.height_m = height_m # This is the motor size (like diameter)

        # Calculate derived properties based on dimensions
        # Assuming width_m is the total span, height_m is the motor diameter/height
        self.motor_radius_m = height_m / 2.0
        # Distance from center to motor center
        self.arm_length_m = (width_m / 2.0) - self.motor_radius_m

        if self.arm_length_m <= 0:
             # Ensure motors don't overlap the frame center if width is too small
             # A simple fix: place motors adjacent to a minimal frame
             self.arm_length_m = self.motor_radius_m
             frame_width_m = 0.1 # Minimal frame width
        else:
             frame_width_m = self.arm_length_m * 2.0

        frame_height_m = height_m * 0.5 # Frame height can be smaller

        # --- Frame ---
        # Note: Pymunk's create_box uses half-width/half-height, but size=(w,h) expects full width/height
        self.frame_shape = pymunk.Poly.create_box(None, size=(frame_width_m, frame_height_m))
        frame_moment = pymunk.moment_for_poly(mass_f, self.frame_shape.get_vertices())
        frame_body = pymunk.Body(mass_f, frame_moment, body_type=pymunk.Body.DYNAMIC)
        frame_body.position = x, y
        frame_body.angle = angle
        self.frame_shape.body = frame_body
        self.frame_shape.color = pygame.Color((66, 135, 245))
        space.add(frame_body, self.frame_shape)
        self.body = frame_body # Reference to the main body

        # --- Left Motor ---
        # Motors are modeled as squares/circles for simplicity
        motor_size = (height_m, height_m)
        self.left_motor_shape = pymunk.Poly.create_box(None, size=motor_size)
        left_motor_moment = pymunk.moment_for_poly(mass_l, self.left_motor_shape.get_vertices())
        left_motor_body = pymunk.Body(mass_l, left_motor_moment, body_type=pymunk.Body.DYNAMIC)
        # Position relative to frame center in world coords requires rotation
        left_rel_pos = Vec2d(-self.arm_length_m, 0).rotated(angle)
        left_motor_body.position = frame_body.position + left_rel_pos
        left_motor_body.angle = angle
        self.left_motor_shape.body = left_motor_body
        self.left_motor_shape.color = pygame.Color((33, 93, 191))
        space.add(left_motor_body, self.left_motor_shape)

        # --- Right Motor ---
        self.right_motor_shape = pymunk.Poly.create_box(None, size=motor_size)
        right_motor_moment = pymunk.moment_for_poly(mass_r, self.right_motor_shape.get_vertices())
        right_motor_body = pymunk.Body(mass_r, right_motor_moment, body_type=pymunk.Body.DYNAMIC)
        # Position relative to frame center
        right_rel_pos = Vec2d(self.arm_length_m, 0).rotated(angle)
        right_motor_body.position = frame_body.position + right_rel_pos
        right_motor_body.angle = angle
        self.right_motor_shape.body = right_motor_body
        self.right_motor_shape.color = pygame.Color((33, 93, 191))
        space.add(right_motor_body, self.right_motor_shape)

        # --- Joints ---
        # Connect motors rigidly to the frame body using two pivot joints per motor
        # Local anchor points relative to each body's center of gravity
        # Left motor joints
        # Joint 1: Center of motor to corresponding point on arm
        pj_l1 = pymunk.PivotJoint(left_motor_body, frame_body, (0, 0), (-self.arm_length_m, 0))
        # Joint 2: Offset point on motor to corresponding offset point on arm (for rigidity)
        # Using a slight vertical offset to prevent perfect alignment if needed
        offset_y = height_m * 0.1
        pj_l2 = pymunk.PivotJoint(left_motor_body, frame_body, (0, offset_y), (-self.arm_length_m, offset_y))
        pj_l1.error_bias = 0 # Make joints very stiff
        pj_l2.error_bias = 0
        space.add(pj_l1, pj_l2)

        # Right motor joints
        pj_r1 = pymunk.PivotJoint(right_motor_body, frame_body, (0, 0), (self.arm_length_m, 0))
        pj_r2 = pymunk.PivotJoint(right_motor_body, frame_body, (0, offset_y), (self.arm_length_m, offset_y))
        pj_r1.error_bias = 0
        pj_r2.error_bias = 0
        space.add(pj_r1, pj_r2)

        # Store references needed by the environment
        self.motor_bodies = [left_motor_body, right_motor_body] # Useful if applying thrust directly to motors
        self.shapes = [self.frame_shape, self.left_motor_shape, self.right_motor_shape]

        # Keep track of the original arm length for applying thrust
        self.thrust_arm_length = self.arm_length_m


    # Optional: Method to apply thrust, assuming thrust comes from motor locations relative to frame
    def apply_thrust(self, left_thrust_force, right_thrust_force):
        # Apply force at the motor attachment points on the FRAME body, in the frame's local up direction
        # This simplifies things as forces/torques are applied to the central body
        self.body.apply_force_at_local_point(Vec2d(0, left_thrust_force), (-self.thrust_arm_length, 0))
        self.body.apply_force_at_local_point(Vec2d(0, right_thrust_force), (self.thrust_arm_length, 0))


    # This method might need adjustment if used, ensure all bodies/shapes are handled
    def change_positions(self, x, y, space):
        # Needs careful implementation if used - must move all bodies and reset velocities/forces
        # Get current angle
        angle = self.body.angle

        # Reset main body
        self.body.position = x, y
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0
        self.body.force = (0, 0)
        self.body.torque = 0
        space.reindex_shapes_for_body(self.body)

        # Recalculate and reset motor positions/velocities based on the frame
        left_rel_pos = Vec2d(-self.arm_length_m, 0).rotated(angle)
        self.left_motor_shape.body.position = self.body.position + left_rel_pos
        self.left_motor_shape.body.velocity = (0, 0)
        self.left_motor_shape.body.angular_velocity = 0
        self.left_motor_shape.body.force = (0, 0)
        self.left_motor_shape.body.torque = 0
        self.left_motor_shape.body.angle = angle
        space.reindex_shapes_for_body(self.left_motor_shape.body)

        right_rel_pos = Vec2d(self.arm_length_m, 0).rotated(angle)
        self.right_motor_shape.body.position = self.body.position + right_rel_pos
        self.right_motor_shape.body.velocity = (0, 0)
        self.right_motor_shape.body.angular_velocity = 0
        self.right_motor_shape.body.force = (0, 0)
        self.right_motor_shape.body.torque = 0
        self.right_motor_shape.body.angle = angle
        space.reindex_shapes_for_body(self.right_motor_shape.body)
