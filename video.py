import cv2
import imageio
import numpy as np
import wandb
import PIL


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20, camera_id=0, use_wandb=True):
        if root_dir is not None:
            self.save_dir = root_dir / "eval_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.context = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, env, enabled=True):
        self.frames = []
        self.context = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, "physics"):
                frame = env.physics.render(
                    height=self.render_size,
                    width=self.render_size,
                    camera_id=self.camera_id,
                )
            else:
                frame = env.render()
            self.frames.append(frame)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 1
        wandb.log(
            {
                "eval/video": wandb.Video(
                    frames[::skip, :, ::2, ::2], fps=fps, format="gif"
                )
            }
        )

    def render_context(self):
        context = np.transpose(np.array(self.context), (0, 3, 1, 2))
        fps, skip = 6, 1
        wandb.log(
            {
                "eval/context": wandb.Video(
                    context[::skip, :, ::2, ::2], fps=fps, format="gif"
                )
            }
        )

    def render_goal(self, env, state):
        with env.physics.reset_context():
            env.physics.set_state(state.cpu())
        if hasattr(env, "physics"):
            frame = env.physics.render(
                height=self.render_size,
                width=self.render_size,
                camera_id=self.camera_id,
            )
        else:
            frame = env.render()
        wandb.log({"eval/goal": wandb.Image(frame)})

    def add_context_frames(self, env, context):
        for i in range(context.shape[0]):
            with env.physics.reset_context():
                env.physics.set_state(context[i].cpu())
            if hasattr(env, "physics"):
                frame = env.physics.render(
                    height=self.render_size,
                    width=self.render_size,
                    camera_id=self.camera_id,
                )
            else:
                frame = env.render()
            self.context.append(frame)

    def save(self, file_name):
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
