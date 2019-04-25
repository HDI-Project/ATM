from __future__ import absolute_import, division, unicode_literals

import logging
import os
import random
import time
from builtins import map, object
from datetime import datetime, timedelta
from operator import attrgetter

from past.utils import old_div

from atm.constants import TIME_FMT, PartitionStatus
from atm.encoder import MetaData
from atm.method import Method
from atm.utilities import download_data, get_public_ip
from atm.worker import ClassifierError, Worker

LOGGER = logging.getLogger(__name__)


class ATM(object):
    """
    Thiss class is code API instance that allows you to use ATM in your python code.
    """

    LOOP_WAIT = 1

    def __init__(self, db, run_conf, aws_conf, log_conf):
        self.db = db
        self.run_conf = run_conf
        self.aws_conf = aws_conf
        self.log_conf = log_conf

    def work(self, datarun_ids=None, save_files=False, choose_randomly=True,
             cloud_mode=False, total_time=None, wait=True):
        """
        Check the ModelHub database for unfinished dataruns, and spawn workers to
        work on them as they are added. This process will continue to run until it
        exceeds total_time or is broken with ctrl-C.

        datarun_ids (optional): list of IDs of dataruns to compute on. If None,
            this will work on all unfinished dataruns in the database.
        choose_randomly: if True, work on all highest-priority dataruns in random
            order. If False, work on them in sequential order (by ID)
        cloud_mode: if True, save processed datasets to AWS. If this option is set,
            aws_config must be supplied.
        total_time (optional): if set to an integer, this worker will only work for
            total_time seconds. Otherwise, it will continue working until all
            dataruns are complete (or indefinitely).
        wait: if True, once all dataruns in the database are complete, keep spinning
            and wait for new runs to be added. If False, exit once all dataruns are
            complete.
        """
        start_time = datetime.now()
        public_ip = get_public_ip()

        # main loop
        while True:
            # get all pending and running dataruns, or all pending/running dataruns
            # from the list we were given
            dataruns = self.db.get_dataruns(include_ids=datarun_ids, ignore_complete=True)
            if not dataruns:
                if wait:
                    LOGGER.warning('No dataruns found. Sleeping %d seconds and trying again.',
                                   ATM.LOOP_WAIT)
                    time.sleep(ATM.LOOP_WAIT)
                    continue

                else:
                    LOGGER.warning('No dataruns found. Exiting.')
                    break

            max_priority = max([datarun.priority for datarun in dataruns])
            priority_runs = [r for r in dataruns if r.priority == max_priority]

            # either choose a run randomly, or take the run with the lowest ID
            if choose_randomly:
                run = random.choice(priority_runs)
            else:
                run = sorted(dataruns, key=attrgetter('id'))[0]

            # say we've started working on this datarun, if we haven't already
            self.db.mark_datarun_running(run.id)

            LOGGER.info('Computing on datarun %d' % run.id)
            # actual work happens here
            worker = Worker(self.db, run, save_files=save_files,
                            cloud_mode=cloud_mode, aws_config=self.aws_conf,
                            log_config=self.log_conf, public_ip=public_ip)
            try:
                worker.run_classifier()

            except ClassifierError:
                # the exception has already been handled; just wait a sec so we
                # don't go out of control reporting errors
                LOGGER.warning('Something went wrong. Sleeping %d seconds.', ATM.LOOP_WAIT)
                time.sleep(ATM.LOOP_WAIT)

            elapsed_time = (datetime.now() - start_time).total_seconds()
            if total_time is not None and elapsed_time >= total_time:
                LOGGER.warning('Total run time for worker exceeded; exiting.')
                break

    def create_dataset(self):
        """
        Create a dataset and add it to the ModelHub database.
        """
        # download data to the local filesystem to extract metadata
        train_local, test_local = download_data(self.run_conf.train_path,
                                                self.run_conf.test_path,
                                                self.aws_conf)

        # create the name of the dataset from the path to the data
        name = os.path.basename(train_local)
        name = name.replace("_train.csv", "").replace(".csv", "")

        # process the data into the form ATM needs and save it to disk
        meta = MetaData(self.run_conf.class_column, train_local, test_local)

        # enter dataset into database
        dataset = self.db.create_dataset(name=name,
                                         description=self.run_conf.data_description,
                                         train_path=self.run_conf.train_path,
                                         test_path=self.run_conf.test_path,
                                         class_column=self.run_conf.class_column,
                                         n_examples=meta.n_examples,
                                         k_classes=meta.k_classes,
                                         d_features=meta.d_features,
                                         majority=meta.majority,
                                         size_kb=old_div(meta.size, 1000))
        return dataset

    def create_datarun(self, dataset):
        """
        Given a config, creates a set of dataruns for the config and enters them into
        the database. Returns the ID of the created datarun.

        dataset: Dataset SQLAlchemy ORM object
        """
        # describe the datarun by its tuner and selector
        run_description = '__'.join([self.run_conf.tuner, self.run_conf.selector])

        # set the deadline, if applicable
        deadline = self.run_conf.deadline
        if deadline:
            deadline = datetime.strptime(deadline, TIME_FMT)
            # this overrides the otherwise configured budget_type
            # TODO: why not walltime and classifiers budget simultaneously?
            self.run_conf.budget_type = 'walltime'
        elif self.run_conf.budget_type == 'walltime':
            deadline = datetime.now() + timedelta(minutes=self.run_conf.budget)

        target = self.run_conf.score_target + '_judgment_metric'
        datarun = self.db.create_datarun(dataset_id=dataset.id,
                                         description=run_description,
                                         tuner=self.run_conf.tuner,
                                         selector=self.run_conf.selector,
                                         gridding=self.run_conf.gridding,
                                         priority=self.run_conf.priority,
                                         budget_type=self.run_conf.budget_type,
                                         budget=self.run_conf.budget,
                                         deadline=deadline,
                                         metric=self.run_conf.metric,
                                         score_target=target,
                                         k_window=self.run_conf.k_window,
                                         r_minimum=self.run_conf.r_minimum)
        return datarun

    def enter_data(self, run_per_partition=False):
        """
        Generate a datarun, including a dataset if necessary.

        Returns: ID of the generated datarun
        """
        # connect to the database

        # if the user has provided a dataset id, use that. Otherwise, create a new
        # dataset based on the arguments we were passed.
        if self.run_conf.dataset_id is None:
            dataset = self.create_dataset()
            self.run_conf.dataset_id = dataset.id
        else:
            dataset = self.db.get_dataset(self.run_conf.dataset_id)

        method_parts = {}
        for m in self.run_conf.methods:
            # enumerate all combinations of categorical variables for this method
            method = Method(m)
            method_parts[m] = method.get_hyperpartitions()
            LOGGER.info('method %s has %d hyperpartitions' %
                        (m, len(method_parts[m])))

        # create hyperpartitions and datarun(s)
        run_ids = []
        if not run_per_partition:
            LOGGER.debug('saving datarun...')
            datarun = self.create_datarun(dataset)

        LOGGER.debug('saving hyperpartions...')
        for method, parts in list(method_parts.items()):
            for part in parts:
                # if necessary, create a new datarun for each hyperpartition.
                # This setting is useful for debugging.
                if run_per_partition:
                    datarun = self.create_datarun(dataset)
                    run_ids.append(datarun.id)

                # create a new hyperpartition in the database
                self.db.create_hyperpartition(datarun_id=datarun.id,
                                              method=method,
                                              tunables=part.tunables,
                                              constants=part.constants,
                                              categoricals=part.categoricals,
                                              status=PartitionStatus.INCOMPLETE)

        LOGGER.info('Data entry complete. Summary:')
        LOGGER.info('\tDataset ID: %d', dataset.id)
        LOGGER.info('\tTraining data: %s', dataset.train_path)
        LOGGER.info('\tTest data: %s', (dataset.test_path or 'None'))

        if run_per_partition:
            LOGGER.info('\tDatarun IDs: %s', ', '.join(map(str, run_ids)))

        else:
            LOGGER.info('\tDatarun ID: %d', datarun.id)

        LOGGER.info('\tHyperpartition selection strategy: %s', datarun.selector)
        LOGGER.info('\tParameter tuning strategy: %s', datarun.tuner)
        LOGGER.info('\tBudget: %d (%s)', datarun.budget, datarun.budget_type)

        return run_ids or datarun.id
