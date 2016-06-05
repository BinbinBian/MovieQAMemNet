import tensorflow as tf
import numpy as np
import skipthoughts.skipthoughts as skip
from IPython import embed


'''
Modification of Memery Network for multiple choice QA
- Data consists of story, question and answers
- Each sentence is encoded into a vector using Skip-Thought Vectors
  (https://github.com/ryankiros/skip-thoughts)
- Encoded vectors are linearly projected by matrix T

'''


class MemN2N(object):
  def __init__(self, config, sess):
    self.init_std = config.init_std
    self.batch_size = config.batch_size
    self.nhop = config.nhop
    self.idim = config.idim     # input vector dimension (=skip-though vector)
    self.edim = config.edim     # encoding dimension
    self.nstory = config.nstory
    self.nanswer = config.nanswer

    self.story = tf.placeholder(tf.float32,
                                [self.batch_size, self.nstory, self.idim],
                                name="story")
    self.query = tf.placeholder(tf.float32,
                                [self.batch_size, 1, self.idim],
                                name="query")
    self.answer = tf.placeholder(tf.float32,
                                 [self.batch_size, self.nanswer, self.idim],
                                 name="answer")
    self.target = tf.placeholder(tf.int64,
                                 [self.batch_size],
                                 name="target")

    self.lr = tf.Variable(config.init_lr)
    self.sess = sess

  def build_memory(self):
    self.global_step = tf.Variable(0, name="global_step")

    # Linear Projection Layer
    self.T = tf.Variable(tf.random_normal([self.idim, self.edim],
                                          stddev=self.init_std,
                                          name="projection"))

    reshape = tf.reshape(self.story, [-1, self.idim])
    m = tf.matmul(reshape, self.T)   # [batch_size * nstory, edim]
    m = tf.reshape(m, [self.batch_size, self.nstory, -1])

    reshape = tf.reshape(self.query, [-1, self.idim])
    u = tf.matmul(reshape, self.T)   # [batch_size * 1, edim]
    u = tf.reshape(u, [self.batch_size, 1, -1])

    reshape = tf.reshape(self.answer, [-1, self.idim])
    g = tf.matmul(reshape, self.T)  # [batch_size * nanswer, edim]
    g = tf.reshape(g, [self.batch_size, self.nanswer, -1])

    for h in xrange(self.nhop):
      p = tf.batch_matmul(m, u, adj_y=True)  # [batch_size, nstory. 1]
      p = tf.reshape(p, [self.batch_size, -1])
      p = tf.nn.softmax(p)  # [batch_size, nstory]

      reshape = tf.reshape(p, [self.batch_size, -1, 1])
      o = tf.reduce_sum(tf.mul(m, reshape), 1)
      u = tf.add(o, u)

    logits = tf.batch_matmul(g, u, adj_y=True)  # [batch_size, nanswer, 1]
    logits = tf.reshape(logits, [self.batch_size, -1])
    self.logits = logits
    self.probs = tf.nn.softmax(logits)

  def build_model(self):
    self.build_memory()
    # embed()
    self.skip_model = skip.load_model()

    self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.target)
    self.opt = tf.train.GradientDescentOptimizer(self.lr)

    grads = self.opt.compute_gradients(self.loss)
    inc_op = self.global_step.assign_add(1)
    with tf.control_dependencies([inc_op]):
      self.apply_grad_op = self.opt.apply_gradients(grads)

    tf.initialize_all_variables().run()

  def encode(self, inputs):
    story = skip.encode(self.skip_model, inputs.story)
    story = np.asarray(story, dtype=np.float32).reshape([self.batch_size, self.nstory, -1])
    query = skip.encode(self.skip_model, inputs.query)
    query = np.asarray(query, dtype=np.float32).reshape([self.batch_size, 1, -1])
    answer = skip.encode(self.skip_model, inputs.answer)
    answer = np.asarray(answer, dtype=np.float32).reshape([self.batch_size, self.nanswer, -1])
    target = np.asarray(inputs.target, dtype=np.int64).reshape([self.batch_size])
    return story, query, answer, target

  def train(self, inputs):
    story, query, answer, target = self.encode(inputs)
    _, loss = self.sess.run([self.apply_grad_op, self.loss],
                            feed_dict={
                              self.story: story,
                              self.query: query,
                              self.answer: answer,
                              self.target: target})
    print "loss: ", loss

  def test(self, inputs):
    story, query, answer, target = self.encode(inputs)
    loss, probs = self.sess.run([self.loss, self.probs],
                                feed_dict={
                                  self.story: story,
                                  self.query: query,
                                  self.answer: answer,
                                  self.target: target})
    print "loss: ", loss
    print "probs: ", probs


















