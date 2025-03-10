import numpy as np
from collections import deque

from myosuite.utils import gym


class NeuralDelayWrapper(gym.Wrapper):
    def __init__(
            self, 
            env, 
            neural_delay: int=2, 
            b: float=1, 
            gravity_range: tuple[float]=[-9.81,-9.81], 
            gravity_xyz: tuple[float]=[0,0,1], 
            viscosity_range: tuple[float]=[0,0], 
            **kwargs
        ):
        """
        Wrapper that applies neural/action delay to a Gymnasium environment.

        The real environment receives a delayed action drawn from a queue of previous
        actions. A parallel headless copy of the environment is kept in sync and used
        to simulate the efferent-copy/forward-simulation process. The wrapper can also
        inject a forward-predicted action from a model and optionally scale the final
        executed action by b.

        Parameters
        ----------
        env : gym.Wrapper
            The target environment to wrap.
        neural_delay : int, default=2
            Number of steps of action delay. Older commands are replayed in the
            delayed queue.
        b : float, default=1
            Scaling factor applied to the executed delayed action. If b == 0, the
            action is zeroed before stepping the environment.
        gravity_range : tuple[float], default=[-9.81, -9.81]
            Range used to sample the gravity vector components at reset.
        gravity_xyz : tuple[float], default=[0, 0, 1]
            Direction vector used to orient gravity.
        viscosity_range : tuple[float], default=[0, 0]
            Range used to sample fluid viscosity at reset.
        """
        self. env = env
        self.henv = gym.make(env.unwrapped.id, render_mode=None, **kwargs) # headless environment for the efferent copy 

        if neural_delay < 0:
            raise ValueError(f'The value {neural_delay} is not a valid input. \nneural_delay must be greater than or equal to 0.')
        self.neural_delay = neural_delay
        self.b            = b

        # Option randomization parameters
        self.gravity_range = tuple(gravity_range)
        self.gravity_xyz   = np.array(gravity_xyz)
        self.gen_force   = lambda: np.random.uniform(self.gravity_range[0], self.gravity_range[1])

        self.viscosity_range = tuple(viscosity_range)
        self.gen_viscosity   = lambda: np.random.uniform(viscosity_range[0], viscosity_range[1])

        # Action buffer stores past actions
        self.action_buffer  = deque([np.zeros(self.env.action_space.shape)], maxlen=neural_delay + 1)
        self.forward_buffer = deque([np.zeros(self.env.action_space.shape)])

    def reset(self, seed=None, model=None, ep_len=None, **kwargs):
        if seed is None:
            self.seed = int(np.random.default_rng().integers(0, 2**32))
        else:
            self.seed = seed

        self. env.unwrapped.seed(self.seed)
        self.henv.unwrapped.seed(self.seed)

        obs, info = self. env.reset(seed=self.seed, **kwargs)
        _, _      = self.henv.reset(seed=self.seed, **kwargs)

        self.action_buffer.clear()
        self.forward_buffer.clear()

        # Fill buffers with initial values
        for _ in range(self.neural_delay + 1):
            self.action_buffer.append(np.zeros(self.env.action_space.shape))

        if model and ep_len:
            self.forward_sim(model, ep_len, obs=obs)
        elif model:
            self.action_buffer.clear()
            self.fill_action_buffer(model, obs=obs)

        if seed is not None:
            np.random.seed(seed)  # Set the seed for reproducibility

        # Update environment options
        x, y, z = self.gen_force(), self.gen_force(), self.gen_force()
        viscosity = self.gen_viscosity()

        self. env.unwrapped.mj_spec.option.gravity[:] = self.gravity_xyz * np.array([x, y, z])
        self.henv.unwrapped.mj_spec.option.gravity[:] = self.gravity_xyz * np.array([x, y, z])

        self. env.unwrapped.mj_spec.option.viscosity = viscosity
        self.henv.unwrapped.mj_spec.option.viscosity = viscosity

        return obs, info  # Return the delayed observation

    def step(self, action, b=None):
        # Store the new action in the buffer
        self.action_buffer.append(action)

        action = self.action_buffer.popleft()
        # Combine the action with the forward buffer if it has values
        if len(self.forward_buffer) > 0:
            action = self.forward_buffer.popleft() # use the forward sim

        # Set b to the default value if not provided
        b = b if b is not None else self.b 

        obs, reward, terminated, truncated, info = self.env.step(action*b)

        # Mentally simulate what will happen
        self.update_headless()
        buffer = self.action_buffer.copy()
        fbuffer = self.forward_buffer.copy()
        for _ in range(self.neural_delay):
            if len(fbuffer) > 0:
                action = (buffer.popleft() + fbuffer.popleft()) * 0.5  # Take the average of the last action and the current one
            else:
                action = buffer.popleft()
            obs, _, _, _, _ = self.henv.step(action)

        # Return the delayed observation (from buffer), with other outputs
        return obs, reward, terminated, truncated, info
    
    def forward_sim(self, model, ep_len, obs=None):
        if obs is None:
            obs, _ = self.henv.reset(seed=self.seed)

        for _ in range(ep_len):
            action, _ = model.predict(obs, deterministic=True, )

            obs, _, _, _, _ = self.henv.step(action)

            self.forward_buffer.append(action)

        return obs

    def fill_action_buffer(self, model=None, obs=None):
        if obs is None:
            obs, _ = self.henv.reset(seed=self.seed)

        for _ in range(self.neural_delay):
            action, _ = model.predict(obs, deterministic=True, )

            obs, _, _, _, _ = self.henv.step(action)

            self.action_buffer.append(action)
    
        return obs
    
    def update_headless(self):
        self.henv.unwrapped.robot.sync_sims(
            self. env.unwrapped.robot.mj_model, self. env.unwrapped.robot.mj_data, # source
            self.henv.unwrapped.robot.mj_model, self.henv.unwrapped.robot.mj_data, # destination
            model=False, data=True
        )

    def close(self):
        self. env.close()
        self.henv.close()
