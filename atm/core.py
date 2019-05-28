# -*- coding: utf-8 -*-

"""Core ATM module.

This module contains the ATM class, which is the one responsible for
executing and orchestrating the main ATM functionalities.
"""

import logging
import random
import time
from datetime import datetime, timedelta
from operator import attrgetter

from tqdm import tqdm

from atm.constants import TIME_FMT, PartitionStatus, RunStatus
from atm.database import Database
from atm.method import Method
from atm.worker import ClassifierError, Worker

LOGGER = logging.getLogger(__name__)


class ATM(object):

    LOOP_WAIT = 5

    def __init__(
        self,

        # SQL Conf
        dialect='sqlite',
        database='atm.db',
        username=None,
        password=None,
        host=None,
        port=None,
        query=None,

        # AWS Conf
        access_key=None,
        secret_key=None,
        s3_bucket=None,
        s3_folder=None,

        # Log Conf
        models_dir='models',
        metrics_dir='metrics',
        verbose_metrics=False,
    ):

        self.db = Database(dialect, database, username, host, port, query)
        self.aws_access_key = access_key
        self.aws_secret_key = secret_key
        self.s3_bucket = s3_bucket
        self.s3_folder = s3_folder

        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self.verbose_metrics = verbose_metrics

    def add_dataset(self, train_path, test_path=None, name=None,
                    description=None, class_column=None):
        return self.db.create_dataset(
            train_path=train_path,
            test_path=test_path,
            name=name,
            description=description,
            class_column=class_column,
            aws_access_key=self.aws_access_key,
            aws_secret_key=self.aws_secret_key,
        )

    def add_datarun(self, dataset_id, budget=100, budget_type='classifier',
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

    def work(self, datarun_ids=None, save_files=True, choose_randomly=True,
             cloud_mode=False, total_time=None, wait=True, verbose=False):
        """
        Check the ModelHub database for unfinished dataruns, and spawn workers to
        work on them as they are added. This process will continue to run until it
        exceeds total_time or is broken with ctrl-C.

        datarun_ids (optional): list of IDs of dataruns to compute on. If None,
            this will work on all unfinished dataruns in the database.
        choose_randomly: if True, work on all highest-priority dataruns in random
            order. If False, work on them in sequential order (by ID)
        cloud_mode: if True, save processed datasets to AWS.
        total_time (optional): if set to an integer, this worker will only work for
            total_time seconds. Otherwise, it will continue working until all
            dataruns are complete (or indefinitely).
        wait: if True, once all dataruns in the database are complete, keep spinning
            and wait for new runs to be added. If False, exit once all dataruns are
            complete.
        """
        start_time = datetime.now()

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

            # either choose a run randomly between priority, or take the run with the lowest ID
            if choose_randomly:
                run = random.choice(dataruns)
            else:
                run = sorted(dataruns, key=attrgetter('id'))[0]

            # say we've started working on this datarun, if we haven't already
            self.db.mark_datarun_running(run.id)

            LOGGER.info('Computing on datarun %d' % run.id)
            # actual work happens here
            worker = Worker(self.db, run, save_files=save_files,
                            cloud_mode=cloud_mode, aws_access_key=self.aws_access_key,
                            aws_secret_key=self.aws_secret_key, s3_bucket=self.s3_bucket,
                            s3_folder=self.s3_folder, models_dir=self.models_dir,
                            metrics_dir=self.metrics_dir, verbose_metrics=self.verbose_metrics)
            try:
                if run.budget_type == 'classifier':
                    pbar = tqdm(
                        total=run.budget,
                        ascii=True,
                        initial=run.completed_classifiers,
                        disable=not verbose
                    )

                    while run.status != RunStatus.COMPLETE:
                        worker.run_classifier()
                        run = self.db.get_datarun(run.id)
                        if verbose and run.completed_classifiers > pbar.last_print_n:
                            pbar.update(run.completed_classifiers - pbar.last_print_n)

                    pbar.close()

                elif run.budget_type == 'walltime':
                    pbar = tqdm(
                        disable=not verbose,
                        ascii=True,
                        initial=run.completed_classifiers,
                        unit=' Classifiers'
                    )

                    while run.status != RunStatus.COMPLETE:
                        worker.run_classifier()
                        run = self.db.get_datarun(run.id)  # Refresh the datarun object.
                        if verbose and run.completed_classifiers > pbar.last_print_n:
                            pbar.update(run.completed_classifiers - pbar.last_print_n)

                    pbar.close()

            except ClassifierError:
                # the exception has already been handled; just wait a sec so we
                # don't go out of control reporting errors
                LOGGER.error('Something went wrong. Sleeping %d seconds.', ATM.LOOP_WAIT)
                time.sleep(ATM.LOOP_WAIT)

            elapsed_time = (datetime.now() - start_time).total_seconds()
            if total_time is not None and elapsed_time >= total_time:
                LOGGER.info('Total run time for worker exceeded; exiting.')
                break

    def run(self, train_path, test_path=None, name=None, description=None,
            class_column='class', budget=100, budget_type='classifier', gridding=0, k_window=3,
            metric='f1', methods=['logreg', 'dt', 'knn'], r_minimum=2, run_per_partition=False,
            score_target='cv', selector='uniform', tuner='uniform', deadline=None, priority=1,
            save_files=True, choose_randomly=True, cloud_mode=False, total_time=None,
            wait=True, verbose=True):

        dataset = self.add_dataset(train_path, test_path, name, description, class_column)
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

        if verbose:
            print('Processing dataset {}'.format(train_path))

        self.work(
            datarun_ids,
            save_files,
            choose_randomly,
            cloud_mode,
            total_time,
            False,
            verbose=verbose
        )

        dataruns = self.db.get_dataruns(
            include_ids=datarun_ids,
            ignore_complete=False,
            ignore_pending=True
        )

        if run_per_partition:
            return dataruns

        elif len(dataruns) == 1:
            return dataruns[0]

    def load_model(self, classifier_id):
        return self.db.get_classifier(classifier_id).load_model()
