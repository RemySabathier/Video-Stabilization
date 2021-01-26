import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk, rectangle
import imageio
import time


class Env():

    def __init__(self, n_shapes=1, size=512, frames=100):
        # Blank image
        self.map = np.zeros((frames, size, size), dtype=np.uint8)
        self.crop_map = np.zeros(
            (frames, int(size/2), int(size/2)), dtype=np.uint8)
        self.jitter_map = np.zeros(
            (frames, int(size/2), int(size/2)), dtype=np.uint8)

        self.size = size
        self.n_shapes = n_shapes
        self.frames = frames

        # Shape dictionnary
        self.shape_dict = {}

        self.jitter_x = []
        self.jitter_y = []

        self.shapes = []

    # Create the dictionnary with all shape parameters
    def _create_shape(self):
        x = np.random.uniform(int(self.size/4), int(3*self.size/4))
        y = np.random.uniform(int(self.size/4), int(3*self.size/4))

        for i in range(self.n_shapes):

            shape_param = {}

            # Speed
            s = np.random.uniform(0, 1)
            theta = np.random.uniform(-np.pi, np.pi)
            # Translation vector
            v = (s*np.cos(theta), s*np.sin(theta))

            # Build the shape
            shape_param['shape'] = 'circle'
            shape_param['r'] = 10
            shape_param['c'] = (x, y)
            shape_param['translation'] = v

            #shape_param['color'] = (255,5,0)

            return shape_param

    def build_frames(self):
        # Initialization: build initial shapes
        for i in range(self.n_shapes):
            self.shape_dict[i] = self._create_shape()

        for z in range(self.frames):

            for elt in self.shape_dict:

                # Compute the center of the shape
                x, y = self.shape_dict[elt]['c']
                x = x + z*self.shape_dict[elt]['translation'][0]
                y = y + z*self.shape_dict[elt]['translation'][1]

                # If center outside the image, remove it
                if (x > self.size-1) or (x < 0) or (y > self.size-1) or (y < 0):
                    self.shape_dict[elt] = self._create_shape()

                else:
                    rr, cc = disk(
                        center=(x, y), radius=self.shape_dict[elt]['r'], shape=(self.size, self.size))
                    # Add the shape to the current frame
                    self.map[z, rr, cc] = 1

    def compute_crop_map(self):
        self.crop_map = self.map[:, self.size //
                                 4:3*self.size//4, self.size//4:3*self.size//4]

    def compute_jittered_map(self):

        self.jitter_x = np.random.randint(20, size=self.frames) - 10
        self.jitter_y = np.random.randint(20, size=self.frames) - 10

        for f in range(self.frames):

            j_x, j_y = self.jitter_x[f], self.jitter_y[f]

            self.jitter_map[f] = self.map[f, j_x+self.size //
                                          4: j_x+3*self.size//4, j_y+self.size//4:j_y+3*self.size//4]

    def get_map(self):
        return self.map

    def make_gif(self, path, fps=30):
        imageio.mimsave('map_'+path, [255*self.map[i]
                                      for i in range(self.frames)], format='GIF', fps=fps)
        imageio.mimsave('cen_'+path, [255*self.crop_map[i]
                                      for i in range(self.frames)], format='GIF', fps=fps)
        imageio.mimsave('jit_'+path, [255*self.jitter_map[i]
                                      for i in range(self.frames)], format='GIF', fps=fps)
        imageio.mimsave(
            'both_'+path, [255*np.concatenate((self.crop_map[i], self.jitter_map[i]), axis=1) for i in range(self.frames)], format='GIF', fps=fps)


if __name__ == '__main__':

    a = time.time()
    env = Env(n_shapes=10, size=512, frames=1000)
    env.build_frames()
    b = time.time()

    print('Frame generated in {:.4f}s'.format(b-a))

    map = env.get_map()

    # Compute cropped and jittered map
    env.compute_crop_map()
    env.compute_jittered_map()

    # Return gif
    env.make_gif('test.gif')

    pass
