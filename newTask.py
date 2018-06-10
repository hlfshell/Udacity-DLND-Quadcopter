import numpy as np
from physics_sim import PhysicsSim

class Task():
    def __init__(self,
                 target_pos,
                 init_pose=None,
                 init_velocities=None,
                 init_angle_velocities=None,
                 runtime=5.,
                 pos_noise=None,
                 ang_noise=None,
                 vel_noise=None,
                 ang_vel_noise=None):
        
        
        self.target_pos = target_pos
        self.pos_noise = pos_noise
        self.ang_noise = ang_noise
        self.vel_noise = vel_noise
        self.ang_vel_noise = ang_vel_noise

        #These are going to get changed a lot
        hover = 400
        
        #Set low/high for actions
        self.action_high = 1.2 * hover
        self.action_low = 0.99 * hover
        
        self.action_size = 1
        
        #Init the velocities to blank defaults if not given to us
        if(init_velocities is None):
            init_velocities = np.array([0.0, 0.0, 0.0])
        if(init_angle_velocities is None):
            init_angle_velocities = np.array([0.0, 0.0, 0.0])

        # Here we create a physics simulation for the task
        # print("start", init_pose, init_velocities, init_angle_velocities)
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)

        self.state_size = len(self.get_state())
        
        #tanh constants
        self.action_b = (self.action_high+self.action_low)/2.0
        self.action_m = (self.action_high-self.action_low)/2.0

    def get_reward(self):
        loss = (self.sim.pose[2]-self.target_pos[2])
        loss = loss ** 2
        loss += 0.1 * (self.sim.linear_accel[2]**2)
        
        delta = 0.5
        reward_max = 1
        reward_min = 0        
        
        #Calulate the Huber Loss
        reward = np.maximum(reward_max - delta * delta * (np.sqrt(1 + (loss / delta) ** 2) - 1), reward_min)
        
        return reward

    def normalize_angles(self, angles):
        # We want to keep angles between
        # -1 and +1
        normalized_angles = np.copy(angles)
        for i in range(len(normalized_angles)):
            while normalized_angles[i] > np.pi:
                normalized_angles[i] -= 2 * np.pi
        return normalized_angles

    def get_state(self):
        position_error = (self.sim.pose[:3] - self.target_pos)

        return np.array([ position_error[2], self.sim.v[2], self.sim.linear_accel[2] ])

    def convert_action(self, action):
        return (action * self.action_m) + self.action_b

    def step(self, action):
        speed_of_rotors = self.convert_action(action)
        is_done = self.sim.next_timestep(speed_of_rotors*np.ones(4))
        next_state = self.get_state()
        reward = self.get_reward()

        if reward <= 0:
            is_done = True

        return next_state, reward, is_done

    def reset(self):
        self.sim.reset()

        #############################################333
        if self.action_size == 1:
            if self.pos_noise is not None or self.ang_noise is not None:
                new_random_pose = np.copy(self.sim.init_pose)
                if self.pos_noise is not None and self.pos_noise > 0:
                    new_random_pose[2] += np.random.normal(0.0, self.pos_noise, 1)

                self.sim.pose = np.copy(new_random_pose)

            if self.vel_noise is not None:
                new_velocity_random = np.copy(self.sim.init_velocities)
                new_velocity_random[2] += np.random.normal(0.0, self.vel_noise, 1)
                self.sim.v = np.copy(new_velocity_random)
            return self.get_state()
        #############################################333
        
        
        #############################################333
        if self.pos_noise is not None or self.ang_noise is not None:
            new_random_pose = np.copy(self.sim.init_pose)
            if self.pos_noise is not None and self.pos_noise > 0:
                new_random_pose[:3] += np.random.normal(0.0, self.pos_noise, 3)
            if self.ang_noise is not None and self.ang_noise > 0:
                new_random_pose[3:] += np.random.normal(0.0, self.ang_noise, 3)

            self.sim.pose = np.copy(new_random_pose)

        #############################################333
        
        #############################################333
        if self.vel_noise is not None:
            new_velocity_random = np.copy(self.sim.init_velocities)
            new_velocity_random += np.random.normal(0.0, self.vel_noise, 3)
            self.sim.v = np.copy(new_velocity_random)
        #############################################333
            
        #############################################333
        if self.ang_vel_noise is not None:
            angle_velocity_random = np.copy(self.sim.init_angle_velocities)
            angle_velocity_random += np.random.normal(0.0, self.ang_vel_noise, 3)
            self.sim.angular_v = np.copy(angle_velocity_random)
        #############################################333
            
        return self.get_state()
