

import time
import os
import logging
import traceback
import json
import glob
import tensorflow as tf
import numpy as np
import h5py


#from clusterone import get_data_path, get_logs_path

from train import *

LOCAL_LOG_LOCATION = "..."

LOCAL_DATASET_NAME = "..."

LOCAL_DATASET_LOCATION = "..."

#Create logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """ Main wrapper"""

    
    try:
        job_name = os.environ['JOB_NAME']
        task_index = os.environ['TASK_INDEX']
        ps_hosts = os.environ['PS_HOSTS']
        worker_hosts = os.environ['WORKER_HOSTS']
    except:
        job_name = None
        task_index = 0
        ps_hosts = None
        worker_hosts = None

    if job_name == None: 
        if LOCAL_LOG_LOCATION == "...":
            raise ValueError("LOCAL_LOG_LOCATION needs to be defined")
        if LOCAL_DATASET_LOCATION == "...":
            raise ValueError("LOCAL_DATASET_LOCATION needs to be defined")
        if LOCAL_DATASET_NAME == "...":
            raise ValueError("LOCAL_DATASET_NAME needs to be defined")

    PATH_TO_LOCAL_LOGS = os.path.expanduser(LOCAL_LOG_LOCATION)
    ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser(LOCAL_DATASET_LOCATION)



    
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    
    flags.DEFINE_string(
        "train_data_dir",
        get_data_path(
            dataset_name = "tensorbot/*",
            local_root = ROOT_PATH_TO_LOCAL_DATA,
            local_repo = LOCAL_DATASET_NAME, #all repos (we use glob downstream, see read_data.py)
            path = 'camera/training/*.h5'#all .h5 files
            ),
        """Path to training dataset. It is recommended to use get_data_path()
        to define your data directory. If you set your dataset directory manually make sure to use /data/
        as root path when running on TensorPort cloud.
        On tensrport, the data will be mounted in /data/user/clusterone_dataset_name,
        so you can acces `path` with  /data/user/clusterone_dataset_name/path
        """
        )
    flags.DEFINE_string("logs_dir",
        get_logs_path(root=PATH_TO_LOCAL_LOGS),
        "Path to store logs and checkpoints. It is recommended"
        "to use get_logs_path() to define your logs directory."
        "If you set your logs directory manually make sure"
        "to use /logs/ when running on TensorPort cloud.")

    # Define worker specific environment variables. Handled automatically.
    flags.DEFINE_string("job_name", job_name,
                        "job name: worker or ps")
    flags.DEFINE_integer("task_index", task_index,
                        "Worker task index, should be >= 0. task_index=0 is "
                        "the chief worker task the performs the variable "
                        "initialization")
    flags.DEFINE_string("ps_hosts", ps_hosts,
                        "Comma-separated list of hostname:port pairs")
    flags.DEFINE_string("worker_hosts", worker_hosts,
                        "Comma-separated list of hostname:port pairs")

    # end of clusterone snippet 2

    # Training flags - feel free to play with that!
    flags.DEFINE_integer("BATCH_SIZE", 64, "Batch size")
    flags.DEFINE_integer("ITERATIONS", 10000, "Iterations per epochs")
    flags.DEFINE_integer("NUM_EPOCHS", 200, "Number of epochs")

    flags.DEFINE_bool("Train" , True , "Training Flag for batch normalization")

    # clusterone snippet 3: configure distributed environment
    def device_and_target():
        # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
        # Don't set a device.
        if FLAGS.job_name is None:
            print("Running single-machine training")
            return (None, "")

        # Otherwise we're running distributed TensorFlow.
        print("Running distributed training")
        if FLAGS.task_index is None or FLAGS.task_index == "":
            raise ValueError("Must specify an explicit `task_index`")
        if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
            raise ValueError("Must specify an explicit `ps_hosts`")
        if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
            raise ValueError("Must specify an explicit `worker_hosts`")

        cluster_spec = tf.train.ClusterSpec({
                "ps": FLAGS.ps_hosts.split(","),
                "worker": FLAGS.worker_hosts.split(","),
        })
        server = tf.train.Server(
                cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        if FLAGS.job_name == "ps":
            server.join()

        worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
        # The device setter will automatically place Variables ops on separate
        # parameter servers (ps). The non-Variable ops will be placed on the workers.
        return (
                tf.train.replica_device_setter(
                        worker_device=worker_device,
                        cluster=cluster_spec),
                server.target,
        )

    device, target = device_and_target()
    # end of clusterone snippet 3

    print(FLAGS.logs_dir)
    print(FLAGS.train_data_dir)

    if FLAGS.logs_dir is None or FLAGS.logs_dir == "":
        raise ValueError("Must specify an explicit `logs_dir`")
    if FLAGS.train_data_dir is None or FLAGS.train_data_dir == "":
        raise ValueError("Must specify an explicit `train_data_dir`")
    # if FLAGS.val_data_dir is None or FLAGS.val_data_dir == "":
    #     raise ValueError("Must specify an explicit `val_data_dir`")


    # Define graph
    with tf.device(device):
        if FLAGS.task_index == 0:
            print("Looking for data in %s" % FLAGS.train_data_dir)
        x , y = read_example(FLAGS.train_data_dir, FLAGS.BATCH_SIZE)

        predictions = yolo_net(x)
        loss = yolo_loss(predictions, y)
        training_summary = tf.summary.scalar('Training_Loss', loss)#add to tboard

        #Batch generators
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.starter_lr, global_step,1000, 0.96, staircase=True)

        train_step = (
            tf.train.AdamOptimizer(learning_rate)
            .minimize(loss, global_step=global_step)
            )

    def run_train_epoch(target,FLAGS,epoch_index):
        """Restores the last checkpoint and runs a training epoch
        Inputs:
            - target: device setter for distributed work
            - FLAGS:
                - requires FLAGS.logs_dir from which the model will be restored.
                Note that whatever most recent checkpoint from that directory will be used.
                - requires FLAGS.steps_per_epoch
            - epoch_index: index of current epoch
        """

        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.steps_per_epoch*epoch_index)] # Increment number of required training steps
        i = 1

        with tf.train.MonitoredTrainingSession(master=target, is_chief = (FLAGS.task_index == 0) , checkpoint_dir = FLAGS.logs_dir , hooks = hooks) as sess:

            while not sess.should_stop():
                variables = [loss, learning_rate, train_step]
                current_loss, lr, _ = sess.run(variables)

                print("Iteration %s - Batch loss: %s" % ((epoch_index)*FLAGS.steps_per_epoch + i,current_loss))
                i+=1

    for e in range(FLAGS.nb_epochs):
        run_train_epoch(target, FLAGS, e)



if __name__ == "__main__":
    main()
