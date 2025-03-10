import myoreachers
from myosuite.utils import gym

from unstablefeedback.neural_delays import NeuralDelayWrapper

def main(render: bool=False):
    steptime = .005 # seconds # 1ms
    episodeLen = 2 # seconds

    env = NeuralDelayWrapper(
        gym.make(
            "myoreachersWristRandom-v0", 
            max_episode_steps=int(episodeLen/steptime),
        ), 
        neural_delay=int(.09/steptime), 
        gravity_range=[0, 0],
        gravity_xyz=[0, 0, 0], 
    )  

    if render:
        env.env.unwrapped.viewer_setup(distance=.3, azimuth=15, elevation=-40)#, lookat=[-1, -1, 0]
        env.unwrapped.mujoco_render_frames = True
        # env.henv.unwrapped.mujoco_render_frames = True

    env     .unwrapped.mj_model.opt.timestep = steptime * .1
    env.henv.unwrapped.mj_model.opt.timestep = steptime * .1

    _ = env.reset()

    for _ in range(400):
        env.step(env.unwrapped.action_space.sample())

    print("Successfully tested the wrapper.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A sample script processing arguments.")
    
    parser.add_argument("-r", "--render", action='store_true', help="Render the reality simulation.")
    
    args = parser.parse_args()
    
    # print(args)

    main(render=args.render)