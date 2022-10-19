import os
import atexit
from typing import Callable

import gym
from gym import logger

from moviepy.editor import ImageSequenceClip, VideoClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip


class VideoRecorder(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            fps: int = 25,
            video_folder: str = './outputs/Videos',
            episode_trigger: Callable[[int], bool] = lambda x: True,
            name_prefix: str = "env_recording",
            max_stored_frames: int = 100,
    ):
        """Wrapper is based off 'RecordVideo' wrapper from gym, see for more details: https://github.com/openai/gym/blob/master/gym/wrappers/record_video.py

        Wrapper collects RGB image arrays in a list at every step, once the list is full (len > max) or the process ends collected frames are converted to video using Moviepy.

        Args:
            env: The environment that will be wrapped
            fps (int): The frames per second the video will played back in.
            video_folder (str): The folder where the recordings will be stored
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            name_prefix (str): Will be prepended to the filename of the recordings
        """

        super().__init__(env)
        self.env = env
        self.fps = fps
        self.video_folder = video_folder
        self.episode_trigger = episode_trigger
        self.name_prefix = name_prefix
        self.max_stored_frames = max_stored_frames  # max frames stored in list before being pushed to video.

        self.video_frames = []  # list of frames to turn into a video clip.
        self.stored_clips = []
        self.recording = False  # bool to determine if frames should be saved or not.
        self.episode_id = 0
        self.file_created_id = 0  # counts the number of times we write video to file to ensure nothing is overwritten.

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `VideoRecorder` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        atexit.register(self.close_video_recorder)  # Function to call At Exit

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if self.recording:
            self.capture_frame()

        if len(self.video_frames) > self.max_stored_frames:
            print('Storage buffer full, making clip')
            self.make_clip()

        if done:
            self.episode_id += 1

        return observation, reward, done, info

    def reset(self):
        if self.episode_trigger(self.episode_id):
            self.recording = True
        else:
            self.recording = False

        observation = self.env.reset()

        return observation

    def capture_frame(self):
        """pull frames from env and store in our video_frames list"""
        if self.env.render_mode is None:
            frame = self.env.render(mode="rgb_array")
        else:
            frame = self.env.render()

        assert type(frame) is list, 'Ensure that the output of render() is a list of RGB images: [np.array()]'

        self.video_frames = self.video_frames + frame

    def make_clip(self):
        """convert all stored frames to a video clip and store this clip in a list, clip is not yet written to file"""
        clip = ImageSequenceClip(self.video_frames, fps=self.fps)
        self.stored_clips.append(clip)
        self.video_frames = []

    def write_clips_to_file(self):
        """write all stored frames and clips to video"""

        if len(self.video_frames) > 0:  # make new clip from remaining frames if exist
            self.make_clip()

        video = CompositeVideoClip(self.stored_clips)
        video.write_videofile(self.video_folder + '/' + self.name_prefix + str(self.file_created_id) + '.mp4')
        self._refresh_buffer()  # probably not necessary

        self.file_created_id += 1

    def _refresh_buffer(self):
        del self.stored_clips, self.video_frames
        self.stored_clips, self.video_frames = [], []

    def close_video_recorder(self):
        """Tidy up for close, write clips to file if remaining"""
        self.recording = False

        if len(self.stored_clips) > 0 or len(self.video_frames) > 0:
            self.write_clips_to_file()

    def close(self):
        """Closes the wrapper."""
        super().close()
        self.close_video_recorder()

    def make_video_from_VideoClip_method(self, agent, duration):

        outer_self = self

        class VideoRun:
            def __init__(self, agent):
                self.agent = agent
                self.done = True

            def loop(self):
                if self.done:
                    self.observation = outer_self.reset()
                action = self.agent.compute_single_action(self.observation)
                self.observation, reward, self.done, _ = outer_self.step(action)

            def make_frame(self, t):
                if not outer_self.video_frames:
                    self.loop()  # only run loop if there are no frames collected

                frame = outer_self.video_frames[0]  # get first frame from list
                outer_self.video_frames = outer_self.video_frames[1:]  # remove this frame
                return frame

        vid = VideoRun(agent)
        func_to_provide_single_frame = vid.make_frame
        clip = VideoClip(func_to_provide_single_frame, duration=duration)
        clip.write_videofile(self.video_folder + '/' + self.name_prefix + str(self.file_created_id) + '.mp4', self.fps)

