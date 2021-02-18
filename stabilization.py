import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import EuclideanTransform, rotate, AffineTransform, warp
import imageio

# Create a video stabilization Model
# Take input all the frames
# need a method compute(prev_frame,next_frame)
# return the stabilized frames


class Stabilizer():

    def __init__(self, frames):

        self.frames = frames
        self.n_frames = frames.shape[0]
        self.corrected_frames = np.zeros_like(frames)

        # The jitters array that will contain the correction
        self.jitter_x = np.zeros((self.n_frames-1,))
        self.jitter_y = np.zeros((self.n_frames-1,))
        self.jitter_r = np.zeros((self.n_frames-1,))

    def step_correction(self, current_step):

        prev_frame = self.frames[current_step]
        next_frame = self.frames[current_step+1]

        # 1 Find good points
        good_points = cv2.goodFeaturesToTrack(
            prev_frame,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3,
        )

        # 2 Compute optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame, next_frame, good_points, None)

        # 3 Extract the current transformation
        idx = np.where(status == 1)[0]
        good_points = good_points[idx]
        curr_pts = curr_pts[idx]

        m, _ = cv2.estimateAffine2D(
            good_points,
            curr_pts,
        )

        # translation
        self.jitter_x[current_step] = m[0, 2]
        self.jitter_y[current_step] = m[1, 2]
        self.jitter_r[current_step] = np.arctan2(m[1, 0], m[0, 0])*180/np.pi

    def compute_correction(self):

        for i in range(self.n_frames-1):
            self.step_correction(i)

        self.jitter_x = np.cumsum(self.jitter_x)
        self.jitter_y = np.cumsum(self.jitter_y)
        self.jitter_r = np.cumsum(self.jitter_r)

        return {'X': self.jitter_x, 'Y': self.jitter_y, 'R': self.jitter_r}

    def apply_correction(self):

        # The original frame is the same
        self.corrected_frames[0] = self.frames[0]

        for i in range(1, self.n_frames):

            j_x = self.jitter_x[i-1]
            j_y = self.jitter_y[i-1]
            j_r = self.jitter_r[i-1]*np.pi/180

            # The inverse transformation matrix
            inv_matrix = np.array([
                [np.cos(j_r), -np.sin(j_r), j_x],
                [np.sin(j_r),  np.cos(j_r), j_y],
                [0, 0, 1],
            ])

            # Apply the inverse transformation to stabilize the image
            tform = AffineTransform(matrix=inv_matrix)

            self.corrected_frames[i] = 255.*warp(self.frames[i], tform)

    def make_gif(self, path, fps=24, start=0):

        imageio.mimsave(
            'corrected_'+path, [np.concatenate((self.frames[i], self.corrected_frames[i]), axis=1) for i in range(start, self.n_frames)], format='GIF', fps=fps)
