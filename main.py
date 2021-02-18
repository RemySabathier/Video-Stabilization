import numpy as np
import matplotlib.pyplot as plt
from video_generator import JitterEnv, FrameBuilder, GaussianJitters, Jitters
from stabilization import Stabilizer
from utils import compute_loss, plot_traj

# Compute the jitters
jt_g = GaussianJitters(scale=0., amplitude_r=0.)
jitters = jt_g.compute_jitters()

# Build random frames
rf = FrameBuilder(n_shapes=0, size=512, frames=1000,
                  min_speed=0.1, max_speed=1, possible_radius=[10, 5, 15])
frames = rf.build_frames()

# Create jittered frames and plot the result
env = JitterEnv(frames, jitters)
env.compute_maps()
# env.make_gif('video_frame.gif', start=800)
# env.plot_jitters('jitters.png')
frames = env.jitter_map


# Stabilize the jittered frames and output result
stab = Stabilizer(frames)
corr = stab.compute_correction()
stab.apply_correction()
stab.make_gif('corrected.gif')

# Compute the stabilization loss
loss = compute_loss(jitters, corr)
print('Loss : {}'.format(loss))

# plot the traj
plot_traj(jitters)
plot_traj(corr, correction=True)

pass
