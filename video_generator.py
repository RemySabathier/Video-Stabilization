import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk, rectangle
import imageio
import time
from skimage.util import img_as_ubyte
from skimage import data


class Env():

    def __init__(self, n_shapes=1, size=512, frames=100, min_speed=1, max_speed=10, possible_radius=[10], min_jitters=-15, max_jitters=15, max_step=2):
        # Blank image

        noisy_image = img_as_ubyte(data.camera())

        self.map = np.zeros((frames, size, size), dtype=np.uint8)

        for i in range(frames):
            self.map[i] = noisy_image

        self.crop_map = np.zeros(
            (frames, int(size/2), int(size/2)), dtype=np.uint8)
        self.jitter_map = np.zeros(
            (frames, int(size/2), int(size/2)), dtype=np.uint8)

        self.size = size
        self.n_shapes = n_shapes
        self.frames = frames

        # Speed parameters
        self.min_speed = min_speed
        self.max_speed = max_speed

        # Radius parameter
        self.possible_radius = possible_radius

        # Jitters parameters
        self.min_jitters = min_jitters
        self.max_jitters = max_jitters
        self.max_step = max_step

        # Shape dictionnary
        self.shape_dict = {}

        self.jitter_x = []
        self.jitter_y = []

        self.shapes = []

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

                # If center outside the image, remove it
                if (x > self.size-1) or (x < 0) or (y > self.size-1) or (y < 0):
                    self.shape_dict[elt] = self._create_shape()

                else:
                    rr, cc = disk(
                        center=(x, y), radius=self.shape_dict[elt]['r'], shape=(self.size, self.size))
                    # Add the shape to the current frame
                    self.map[f, rr, cc] = self.shape_dict[elt]['color']
                    self.shape_dict[elt]['time_step'] += 1

    def compute_crop_map(self):
        self.crop_map = self.map[:, self.size //
                                 4:3*self.size//4, self.size//4:3*self.size//4]

    def compute_jittered_map(self):

        self.jitter_x = np.clip(np.cumsum(np.random.randint(
            self.max_step*2+1, size=self.frames) - 2), a_min=self.min_jitters, a_max=self.max_jitters)
        self.jitter_y = np.clip(np.cumsum(np.random.randint(
            self.max_step*2+1, size=self.frames) - 2), a_min=self.min_jitters, a_max=self.max_jitters)

        for f in range(self.frames):

            j_x, j_y = self.jitter_x[f], self.jitter_y[f]

            self.jitter_map[f] = self.map[f, j_x+self.size //
                                          4: j_x+3*self.size//4, j_y+self.size//4:j_y+3*self.size//4]

    def get_map(self):
        return self.map

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

    start = time.time()

    # Build the environment
    env = Env(
        n_shapes=20,
        size=512,
        frames=1000,
        min_speed=0.1,
        max_speed=1,
        possible_radius=[10, 5, 15],

        min_jitters=-30,
        max_jitters=30,
        max_step=2,
    )
    env.build_frames()
    stop = time.time()

    print('Frame generated in {:.4f}s'.format(stop-start))

    map = env.get_map()

    # Compute cropped and jittered map
    env.compute_crop_map()
    env.compute_jittered_map()

    # Return gif
    env.make_gif('test.gif')

    # Save jitter plot
    env.plot_jitters('jitters.png')

    pass
