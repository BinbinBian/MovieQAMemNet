import tensorflow as tf

from model import MemN2N

flags = tf.app.flags

flags.DEFINE_integer("idim", 4800, "input skip-thought vector dimension")
flags.DEFINE_integer("edim", 150, "encoding dimension")
flags.DEFINE_integer("nhop", 3, "number of hops")
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("nstory", 3, "number of story sentences")
flags.DEFINE_integer("nanswer", 5, "number of answer sentences")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate")
flags.DEFINE_float("init_std", 0.05, "weight initialization std")

FLAGS = flags.FLAGS


class example(object):
  story = ["Kevin went into the kitchen", "Kevin washed dishes", "Kevin went into the bedroom"]
  query = ["What did Kevin do in the kitchen?"]
  answer = ["He cooked his lunch", "He cleaned the floor", "He washed dishes", "He took a walk", "He took out a bowl"]
  target = 3

def main(_):
  with tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
    device_count={'GPU': 1})) as sess:
    model = MemN2N(FLAGS, sess)
    model.build_model()

    model.test(example)

if __name__ == '__main__':
  tf.app.run()
