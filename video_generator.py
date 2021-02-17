import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk, rectangle
import imageio
import time
from skimage.util import img_as_ubyte
from skimage import data
from skimage.transform import EuclideanTransform, rotate

# Class that creates random frames and outputs list of jitters (x,y,r)


class Jitters():
    '''The Jitter class creates random movement to apply to frames'''

    def __init__(self, size=1000, min_jitters=-30, max_jitters=30, max_step=2, min_rotation=-10, max_rotation=10, max_step_rotation=2):

        self.size = size
        self.min_jitters = min_jitters
        self.max_jitters = max_jitters
        self.max_step = max_step
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.max_step_rotation = max_step_rotation

    def compute_jitters(self):
        '''Compute the jitter and output a dictionnary with all the jitter parameters'''
        self.jitter_x = np.clip(np.cumsum(np.random.randint(
            self.max_step*2+1, size=self.size) - 2), a_min=self.min_jitters, a_max=self.max_jitters)

        self.jitter_y = np.clip(np.cumsum(np.random.randint(
            self.max_step*2+1, size=self.size) - 2), a_min=self.min_jitters, a_max=self.max_jitters)

        self.jitter_r = np.clip(np.cumsum(np.random.randint(
            self.max_step_rotation*2+1, size=self.size) - 2), a_min=self.min_rotation, a_max=self.max_rotation)

        return {'X': self.jitter_x, 'Y': self.jitter_y, 'R': self.jitter_r}


# Class that creates frames
class FrameBuilder():
    '''Class that creates frames with a background and some images'''

    def __init__(self,
                 n_shapes=1,
                 size=512,
                 frames=100,
                 min_speed=1,
                 max_speed=10,
                 possible_radius=[10]
                 ):

        self.size = size
        self.n_shapes = n_shapes
        self.frames = frames

        # Speed parameters
        self.min_speed = min_speed
        self.max_speed = max_speed

        # Radius parameter
        self.possible_radius = possible_radius

        # Shape dictionnary
        self.shape_dict = {}

        # Array that will store all the frames
        self.map = np.zeros((frames, size, size), dtype=np.uint8)

        # The background image
        noisy_image = img_as_ubyte(data.camera())
        # White frames
        self.map = np.zeros((frames, size, size), dtype=np.uint8)

        for i in range(frames):
            self.map[i] = noisy_image

    # Create the dictionnary with all shape parameters

    def _create_shape(self):

        a = np.random.uniform(2, self.size-2)

        x, y = [(a, 0), (0, a), (self.size-1, a),
                (a, self.size-1)][np.random.randint(4)]

        shape_param = {}

        # Speed
        s = np.random.uniform(self.min_speed, self.max_speed)

        border_1, border_2 = self.size//4, 3*self.size//4
        L_border_x = [border_1, border_1, border_2, border_2]
        L_border_y = [border_1, border_2, border_1, border_2]

        L_angle = [np.arctan2(l_y-y, l_x-x)
                   for l_x, l_y in zip(L_border_x, L_border_y)]

        min_angle, max_angle = min(L_angle), max(L_angle)

        if x == 0 and y > 0:

            if all([i < 0 for i in L_angle]) or all([i > 0 for i in L_angle]):
                zone = [min_angle, max_angle]
                theta = np.random.uniform(zone[0], zone[1])

            else:
                L_angle_neg = [i for i in L_angle if i < 0]
                L_angle_pos = [i for i in L_angle if i > 0]

                zone1 = [min(L_angle_neg), 0]
                zone2 = [0, max(L_angle_pos)]

                theta = np.random.choice(
                    [np.random.uniform(zone1[0], zone1[1]), np.random.uniform(zone2[0], zone2[1])])

        if (x == self.size-1) and (y > 0):

            if all([i < 0 for i in L_angle]) or all([i > 0 for i in L_angle]):
                zone = [min_angle, max_angle]
                theta = np.random.uniform(zone[0], zone[1])

            else:
                L_angle_neg = [i for i in L_angle if i < 0]
                L_angle_pos = [i for i in L_angle if i > 0]

                zone1 = [-np.pi, max(L_angle_neg)]
                zone2 = [min(L_angle_pos), np.pi]

                theta = np.random.choice(
                    [np.random.uniform(zone1[0], zone1[1]), np.random.uniform(zone2[0], zone2[1])])

        else:
            zone = [min_angle, max_angle]
            theta = np.random.uniform(zone[0], zone[1])

        # Translation vector
        v = (s*np.cos(theta), s*np.sin(theta))

        # Build the shape
        shape_param['shape'] = 'circle'
        shape_param['r'] = np.random.choice(self.possible_radius)
        shape_param['c'] = (x, y)
        shape_param['translation'] = v
        shape_param['time_step'] = 0

        # np.random.randint(0, 255, 1)  # Grayscale
        shape_param['color'] = 255.

        return shape_param

    def build_frames(self):

        # Initialization: build initial shapes
        for i in range(self.n_shapes):
            self.shape_dict[i] = self._create_shape()

        for f in range(self.frames):

            for elt in range(len(self.shape_dict)):

                # Compute the center of the shape
                x, y = self.shape_dict[elt]['c']
                t = self.shape_dict[elt]['time_step']
                x = x + t*self.shape_dict[elt]['translation'][0]
                y = y + t*self.shape_dict[elt]['translation'][1]

                # If center outside the image, replace it by a new shape
                if (x > self.size-1) or (x < 0) or (y > self.size-1) or (y < 0):
                    self.shape_dict[elt] = self._create_shape()

                else:
                    rr, cc = disk(
                        center=(x, y), radius=self.shape_dict[elt]['r'], shape=(self.size, self.size))
                    # Add the shape to the current frame
                    self.map[f, rr, cc] = self.shape_dict[elt]['color']
                    self.shape_dict[elt]['time_step'] += 1

        return {'frames': self.map}


# Class that takes frames and jitters and compute the jittered frames
class JitterEnv():
    '''Takes frames and jitters as parameters and computes the jittered frames'''

    def __init__(self, frames, jitters, crop_ratio=1/4):

        self.map = frames['frames']
        self.frames = self.map.shape[0]
        self.size = self.map.shape[1]

        self.jitter_x = jitters['X']
        self.jitter_y = jitters['Y']
        self.jitter_r = jitters['R']

        self.crop_ratio = crop_ratio

        l1, l2 = int(self.crop_ratio*self.size), int(3 *
                                                     self.crop_ratio*self.size)

        self.jitter_map = np.zeros((self.frames, l2-l1, l2-l1), dtype=np.uint8)
        self.crop_map = np.zeros((self.frames, l2-l1, l2-l1), dtype=np.uint8)

    def compute_crop_map(self):

        l1, l2 = int(self.crop_ratio*self.size), int(3 *
                                                     self.crop_ratio*self.size)
        self.crop_map = self.map[:, l1:l2, l1:l2]

    def compute_jittered_map(self):

        for f in range(self.frames):

            j_x, j_y, j_r = self.jitter_x[f], self.jitter_y[f], self.jitter_r[f]

            # Rotation
            current_map = rotate(self.map[f], angle=j_r)*255.
            current_map = current_map.astype(np.uint8)

            # Translation
            l1, l2 = int(self.crop_ratio*self.size), int(3 *
                                                         self.crop_ratio*self.size)
            self.jitter_map[f] = current_map[j_x+l1:j_x+l2, j_y+l1:j_y+l2]

            pass

    def compute_maps(self):
        self.compute_jittered_map()
        self.compute_crop_map()

    def make_gif(self, path, fps=24):

        imageio.mimsave('map_'+path, [self.map[i]
                                      for i in range(self.frames)], format='GIF', fps=fps)
        # imageio.mimsave('cen_'+path, [255*self.crop_map[i]
        #                              for i in range(self.frames)], format='GIF', fps=fps)
        # imageio.mimsave('jit_'+path, [255*self.jitter_map[i]
        #                              for i in range(self.frames)], format='GIF', fps=fps)
        imageio.mimsave(
            'both_'+path, [np.concatenate((self.crop_map[i], self.jitter_map[i]), axis=1) for i in range(self.frames)], format='GIF', fps=fps)

    def plot_jitters(self, path):
        plt.figure(figsize=(8, 6))
        plt.title('Applied Jittters')
        plt.plot(self.jitter_x, label='Jitter X')
        plt.plot(self.jitter_y, label='Jitter Y')
        plt.legend()
        plt.xlabel('time step')
        plt.ylabel('pixel deviation')
        plt.savefig(path)


if __name__ == '__main__':

    # Compute the jitters
    jt = Jitters()
    jitters = jt.compute_jitters()

    # Build random frames
    rf = FrameBuilder(n_shapes=20, size=512, frames=1000,
                      min_speed=0.1, max_speed=1, possible_radius=[10, 5, 15])
    frames = rf.build_frames()

    # Create jittered frames and plot the result
    env = JitterEnv(frames, jitters)
    env.compute_maps()
    env.make_gif('video_frame.gif')
    env.plot_jitters('jitters.png')

    pass
