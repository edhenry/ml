import torch
import tensorflow as tf
import numpy as np
import scipy.misc

try:
    from StringIO import StringIO #python 2.7
except ImportError:
    from io import BytesIO #python 3.x


class Logger(object):
    
    def __init__(self, log_dir: str):
        """Create a summary writer that logs to log_dir
        
        Arguments:
            log_dir {str} -- location of the directory on local disc to save the log
        """
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag: str, value: int, step: int):
        """Log a scalar variable
        
        Arguments:
            tag {str} -- [description]
            value {int} -- [description]
            step {int} -- [description]
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    # TODO inspect and add typing information for images variable
    def image_summary(self, tag: str, images, step: int):
        """Log a list of images
        
        Arguments:
            tag {str} -- [description]
            images {[type]} -- [description]
            step {int} -- [description]
        """
        img_summaries = []
        
        for i, img in enumerate(images):
            try:
                s = StringIO()
            except:
                s = BytesIO()
            
            scipy.misc.toimage(img).save(s, format='png')

            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])

            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    #TODO inspect and add type information for values array
    def histogram_summary(self, tag: str, values, step: int, bins=1000):
        """Log histogram of the tensor values
        
        Arguments:
            tag {str} -- [description]
            values {[type]} -- [description]
            step {int} -- [description]
        
        Keyword Arguments:
            bins {[type]} -- [description] (default: {1000:int})
        """

        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = float(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()






