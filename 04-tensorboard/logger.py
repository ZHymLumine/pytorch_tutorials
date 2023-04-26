import tensorflow as tf
import numpy as np
import imageio
from io import BytesIO
from PIL import Image


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        with self.writer.as_default():
            for tag, value in tag_value_pairs:
                tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        # self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        with self.writer.as_default():
            # Convert the list of images into a single tensor of shape [num_images, height, width, channels]
            img_tensor = np.stack(images, axis=0)
            img_tensor = np.expand_dims(img_tensor, -1)  # Add a channel dimension if the images are grayscale

            # Create an Image object
            img_summary = tf.summary.image(name=tag, data=img_tensor, step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Create the histogram using bin_edges[:-1] as edges and counts as values
        hist_data = list(zip(bin_edges[:-1], counts))

        # Create and write Summary
        with self.writer.as_default():
            tf.summary.histogram(tag, hist_data, step=step)
            self.writer.flush()


