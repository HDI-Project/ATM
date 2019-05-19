# -*- coding: utf-8 -*-

"""Core ATM module.

This module contains the ATM class, which is the one responsible for
executing and orchestrating the main ATM functionalities.
"""

import logging
import random
import time
from datetime import datetime
from operator import attrgetter

import yaml

from atm.config import AWSConfig, LogConfig, SQLConfig
from atm.constants import TIME_FMT, PartitionStatus
from atm.database import Database
from atm.method import Method
from atm.utilities import get_public_ip
from atm.worker import ClassifierError, Worker

LOGGER = logging.getLogger(__name__)


class ATM:

    LOOP_WAIT = 5

    def __init__(self, dialect='sqlite', database='atm.db', username=None, password=None,
                 host=None, port=None, query=None, access_key=None, secret_key=None,
                 s3_bucket=None, s3_folder=None, config_path=None, model_dir='models',
                 metric_dir='metrics', verbose_metrics=False,
                 sql_conf=None, aws_conf=None, log_conf=None, **kwargs):

        if config_path:
            with open(config_path, 'rb') as f:
                args = yaml.load(f)

            self.db = Database(**SQLConfig(args).to_dict())
            self.aws_conf = AWSConfig(args)
            self.log_conf = LogConfig(args)

        else:
            # create database with dialect / username / password
            if sql_conf:
                self.db = Database(**sql_conf.to_dict())
            else:
                self.db = Database(**SQLConfig(locals()).to_dict())

            if aws_conf:
                self.aws_conf = aws_conf
            else:
                self.aws_conf = AWSConfig(locals())

            if log_conf:
                self.log_conf = log_conf
            else:
                self.log_conf = LogConfig(locals())

    def add_dataset(self, train_path=None, test_path=None, name=None,
                    description=None, class_column=None, dataset_conf=None):

        if dataset_conf:
            return self.db.create_dataset(**dataset_conf.to_dict())

        else:
            return self.db.create_dataset(
                name=name,
                description=description,
                train_path=train_path,
                test_path=test_path,
                class_column=class_column,
                aws_conf=self.aws_conf
            )

    def add_datarun(self, dataset_id=None, budget=100, budget_type='classifier',
                    gridding=0, k_window=3, metric='f1', methods=['logreg', 'dt', 'knn'],
                    r_minimum=2, run_per_partition=False, score_target='cv', priority=1,
                    selector='uniform', tuner='uniform', deadline=None):

        dataruns = list()

        if deadline:
            deadline = datetime.strptime(deadline, TIME_FMT)
            budget_type = 'walltime'

        elif budget_type == 'walltime':
            deadline = datetime.now() + timedelta(minutes=budget)

        run_description = '___'.join([tuner, selector])
        target = score_target + '_judgment_metric'

        method_parts = {}
        for method in methods:
            # enumerate all combinations of categorical variables for this method
            method_instance = Method(method)
            method_parts[method] = method_instance.get_hyperpartitions()

            LOGGER.info('method {} has {} hyperpartitions'.format(
                method, len(method_parts[method])))

        if not run_per_partition:
            datarun = self.db.create_datarun(
                dataset_id=dataset_id,
                description=run_description,
                tuner=tuner,
                selector=selector,
                gridding=gridding,
                priority=priority,
                budget_type=budget_type,
                budget=budget,
                deadline=deadline,
                metric=metric,
                score_target=target,
                k_window=k_window,
                r_minimum=r_minimum
            )

            dataruns.append(datarun)

        for method, parts in method_parts.items():
            for part in parts:
                # if necessary, create a new datarun for each hyperpartition.
                # This setting is useful for debugging.
                if run_per_partition:
                    datarun = self.db.create_datarun(
                        dataset_id=dataset_id,
                        description=run_description,
                        tuner=tuner,
                        selector=selector,
                        gridding=gridding,
                        priority=priority,
                        budget_type=budget_type,
                        budget=budget,
                        deadline=deadline,
                        metric=metric,
                        score_target=target,
                        k_window=k_window,
                        r_minimum=r_minimum
                    )

                    dataruns.append(datarun)

                # create a new hyperpartition in the database
                self.db.create_hyperpartition(datarun_id=datarun.id,
                                              method=method,
                                              tunables=part.tunables,
                                              constants=part.constants,
                                              categoricals=part.categoricals,
                                              status=PartitionStatus.INCOMPLETE)

        dataset = self.db.get_dataset(dataset_id)
        LOGGER.info('Dataruns created. Summary:')
        LOGGER.info('\tDataset ID: {}'.format(dataset.id))
        LOGGER.info('\tTraining data: {}'.format(dataset.train_path))
        LOGGER.info('\tTest data: {}'.format(dataset.test_path))

        if run_per_partition:
            LOGGER.info('\tDatarun IDs: {}'.format(
                ', '.join(str(datarun.id) for datarun in dataruns)))

        else:
            LOGGER.info('\tDatarun ID: {}'.format(dataruns[0].id))

        LOGGER.info('\tHyperpartition selection strategy: {}'.format(dataruns[0].selector))
        LOGGER.info('\tParameter tuning strategy: {}'.format(dataruns[0].tuner))
        LOGGER.info('\tBudget: {} ({})'.format(dataruns[0].budget, dataruns[0].budget_type))

        return dataruns if run_per_partition else dataruns[0]

    def create_dataruns(self, run_conf):
        """
        Generate a datarun, including a dataset if necessary.

        Returns: ID of the generated datarun
        """
        dataset = self.db.get_dataset(run_conf.dataset_id)
        if not dataset:
            raise ValueError('Invalid Dataset ID: {}'.format(run_conf.dataset_id))

        method_parts = {}
        for m in run_conf.methods:
            # enumerate all combinations of categorical variables for this method
            method = Method(m)
            method_parts[m] = method.get_hyperpartitions()
            LOGGER.info('method %s has %d hyperpartitions' % (m, len(method_parts[m])))

        # create hyperpartitions and datarun(s)
        dataruns = []
        if not run_conf.run_per_partition:
            LOGGER.debug('saving datarun...')
            datarun = self.create_datarun(dataset, run_conf)
            dataruns.append(datarun)

        LOGGER.debug('saving hyperpartions...')
        for method, parts in list(method_parts.items()):
            for part in parts:
                # if necessary, create a new datarun for each hyperpartition.
                # This setting is useful for debugging.
                if run_conf.run_per_partition:
                    datarun = self.create_datarun(dataset, run_conf)
                    dataruns.append(datarun)

                # create a new hyperpartition in the database
                self.db.create_hyperpartition(datarun_id=datarun.id,
                                              method=method,
                                              tunables=part.tunables,
                                              constants=part.constants,
                                              categoricals=part.categoricals,
                                              status=PartitionStatus.INCOMPLETE)

        LOGGER.info('Dataruns created. Summary:')
        LOGGER.info('\tDataset ID: %d', dataset.id)
        LOGGER.info('\tTraining data: %s', dataset.train_path)
        LOGGER.info('\tTest data: %s', (dataset.test_path or 'None'))

        datarun = dataruns[0]
        if run_conf.run_per_partition:
            LOGGER.info('\tDatarun IDs: %s', ', '.join(str(datarun.id) for datarun in dataruns))

        else:
            LOGGER.info('\tDatarun ID: %d', datarun.id)

        LOGGER.info('\tHyperpartition selection strategy: %s', datarun.selector)
        LOGGER.info('\tParameter tuning strategy: %s', datarun.tuner)
        LOGGER.info('\tBudget: %d (%s)', datarun.budget, datarun.budget_type)

        return dataruns

    def run(self, train_path=None, test_path=None, name=None, description=None,
            column_name='class', budget=100, budget_type='classifier', gridding=0, k_window=3,
            metric='f1', methods=['logreg', 'dt', 'knn'], r_minimum=2, run_per_partition=False,
            score_target='cv', selector='uniform', tuner='uniform', deadline=None, priority=1,
            save_files=False, choose_randomly=True, cloud_mode=False, total_time=None,
            wait=True, dataset_conf=None, run_conf=None):
        """Returns Datarun."""

        if dataset_conf:
            dataset = self.add_dataset(**dataset_conf.to_dict())

        else:
            dataset = self.add_dataset(train_path, test_path, name, description, column_name)

        if run_conf:
            datarun = self.add_datarun(**run_conf.to_dict())

        else:
            datarun = self.add_datarun(
                dataset.id,
                budget,
                budget_type,
                gridding,
                k_window,
                metric,
                methods,
                r_minimum,
                run_per_partition,
                score_target,
                priority,
                selector,
                tuner,
                deadline
            )

        if run_per_partition:
            datarun_ids = [_datarun.id for _datarun in datarun]

        else:
            datarun_ids = [datarun.id]

        self.work(datarun_ids, save_files, choose_randomly, cloud_mode, total_time, False)

        return datarun

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
                    LOGGER.debug('No dataruns found. Sleeping %d seconds and trying again.',
                                 ATM.LOOP_WAIT)
                    time.sleep(ATM.LOOP_WAIT)
                    continue

                else:
                    LOGGER.info('No dataruns found. Exiting.')
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
                LOGGER.error('Something went wrong. Sleeping %d seconds.', ATM.LOOP_WAIT)
                time.sleep(ATM.LOOP_WAIT)

            elapsed_time = (datetime.now() - start_time).total_seconds()
            if total_time is not None and elapsed_time >= total_time:
                LOGGER.info('Total run time for worker exceeded; exiting.')
                break

    def load_model(self, classifier_id):
        """Returns a Model instance."""
        return self.db.get_classifier(classifier_id).load_model()

    def enter_data(self, dataset_conf, run_conf):
        """
        Generate a datarun, including a dataset if necessary.
        Returns: ID of the generated datarun
        """
        # if the user has provided a dataset id, use that. Otherwise, create a new
        # dataset based on the arguments we were passed.
        if run_conf.dataset_id is None:
            dataset = self.add_dataset(dataset_conf=dataset_conf)
            run_conf.dataset_id = dataset.id

        return self.add_datarun(**run_conf.to_dict())
