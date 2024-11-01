# envs/procgen_env.py
import gym
import numpy as np
import procgen


class ProcGen:
    def __init__(
        self,
        name,
        num_levels=0,
        start_level=0,
        distribution_mode="easy",
        render_mode="rgb_array",
        size=(64, 64),
        seed=0,
    ):
        '''self._env = gym.make(
            "procgen:procgen-" + name + "-v0",
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            render_mode=render_mode,
        )'''
        self._env = gym.make(
            "procgen:procgen-" + name + "-v0",
        )
        self._size = size
        # self._env.seed(seed)
        self.reward_range = [-np.inf, np.inf]
        shape = self._env.observation_space.shape
        self._buffer = np.zeros(shape, np.uint8)
        self._done = True
        self._step = 0
        self._gray = False
        self._length = 108000

    @property
    def observation_space(self):
        obs_space = self._env.observation_space
        img_space = gym.spaces.Box(
            0, 255, (*self._size, obs_space.shape[-1]), dtype=np.uint8
        )
        return gym.spaces.Dict({"image": img_space})

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        total = 0.0
        action = np.argmax(action)
        obs, reward, over, info = self._env.step(action)
        self._step += 1
        total += reward
        self._screen(self._buffer, obs)
        self._done = over or (self._length and self._step >= self._length)
        return self._obs(
            total,
            is_last=self._done,
            is_terminal=over,
        )

    def reset(self):
        obs = self._env.reset()
        self._screen(self._buffer, obs)
        self._step = 0
        self._done = False
        obs, reward, is_terminal, _ = self._obs(obs, is_first=True)
        return obs

    def _screen(self, buffer, obs):
        buffer[:] = obs

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        image = self._buffer
        if image.shape[:2] != self._size:
            if self._resize == "opencv":
                image = self._cv2.resize(
                    image, self._size, interpolation=self._cv2.INTER_AREA
                )
            if self._resize == "pillow":
                image = self._image.fromarray(image)
                image = image.resize(self._size, self._image.NEAREST)
                image = np.array(image)
        if self._gray:
            weights = [0.299, 0.587, 1 - (0.299 + 0.587)]
            image = np.tensordot(image, weights, (-1, 0)).astype(image.dtype)
            image = image[:, :, None]
        return (
            {"image": image, "is_terminal": is_terminal, "is_first": is_first},
            reward,
            is_last,
            {},
        )

    def close(self):
        self._env.close()
