from tensorflow.keras.callbacks import Callback
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.distribute import worker_training_state

class BackupAndRestore(Callback):


  def __init__(self, backup_dir):
    super(BackupAndRestore, self).__init__()
    self.backup_dir = backup_dir
    self._supports_tf_logs = True
    self._supported_strategies = (
        distribute_lib._DefaultDistributionStrategy,
        mirrored_strategy.MirroredStrategy,
        collective_all_reduce_strategy.CollectiveAllReduceStrategy)

    if not context.executing_eagerly():
      if ops.inside_function():
        raise ValueError('This Callback\'s method contains Python state and '
                         'should be called outside of `tf.function`s.')
      else:  # Legacy graph mode:
        raise ValueError(
            'BackupAndRestore only supports eager mode. In graph '
            'mode, consider using ModelCheckpoint to manually save '
            'and restore weights with `model.load_weights()` and by '
            'providing `initial_epoch` in `model.fit()` for fault tolerance.')

    # Only the chief worker writes model checkpoints, but all workers
    # restore checkpoint at on_train_begin().
    self._chief_worker_only = False

  def set_model(self, model):
    self.model = model

  def on_train_begin(self, logs=None):
    # TrainingState is used to manage the training state needed for
    # failure-recovery of a worker in training.
    # pylint: disable=protected-access

    if not isinstance(self.model.distribute_strategy,
                      self._supported_strategies):
      raise NotImplementedError(
          'Currently only support empty strategy, MirroredStrategy and '
          'MultiWorkerMirroredStrategy.')
    self.model._training_state = (
        worker_training_state.WorkerTrainingState(self.model, self.backup_dir))
    self._training_state = self.model._training_state
    self._training_state.restore()

  def on_train_end(self, logs=None):
    # pylint: disable=protected-access
    # On exit of training, delete the training state backup file that was saved
    # for the purpose of worker recovery.
    self._training_state.delete_backup()

    # Clean up the training state.
    del self._training_state
    del self.model._training_state

  def on_epoch_end(self, epoch, logs=None):
    # Back up the model and current epoch for possible future recovery.
    self._training_state.back_up(epoch)
