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

    _LOOP_WAIT = 5

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
        """Add a new dataset to the Database.

        Args:
            train_path (str):
                Path to the training CSV file. It can be a local filesystem path,
                absolute or relative, or an HTTP or HTTPS URL, or an S3 path in the
                format ``s3://{bucket_name}/{key}``. Required.
            test_path (str):
                Path to the testing CSV file. It can be a local filesystem path,
                absolute or relative, or an HTTP or HTTPS URL, or an S3 path in the
                format ``s3://{bucket_name}/{key}``.
                Optional. If not given, the training CSV will be split in two parts,
                train and test.
            name (str):
                Name given to this dataset. Optional. If not given, a hash will be
                generated from the training_path and used as the Dataset name.
            description (str):
                Human friendly description of the Dataset. Optional.
            class_column (str):
                Name of the column that will be used as the target variable.
                Optional. Defaults to ``'class'``.

        Returns:
            Dataset:
                The created dataset.
        """

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

        """Register one or more Dataruns to the Database.

        The methods hyperparameters will be analyzed and Hyperpartitions generated
        from them.
        If ``run_per_partition`` is ``True``, one Datarun will be created for each
        Hyperpartition. Otherwise, a single one will be created for all of them.

        Args:
            dataset_id (int):
                Id of the Dataset which this Datarun will belong to.
            budget (int):
                Budget amount. Optional. Defaults to ``100``.
            budget_type (str):
                Budget Type. Can be 'classifier' or 'walltime'.
                Optional. Defaults to ``'classifier'``.
            gridding (int):
                ``gridding`` setting for the Tuner. Optional. Defaults to ``0``.
            k_window (int):
                ``k`` setting for the Selector. Optional. Defaults to ``3``.
            metric (str):
                Metric to use for the tuning and selection. Optional. Defaults to ``'f1'``.
            methods (list):
                List of methods to try. Optional. Defaults to ``['logreg', 'dt', 'knn']``.
            r_minimum (int):
                ``r_minimum`` setting for the Tuner. Optional. Defaults to ``2``.
            run_per_partition (bool):
                whether to create a separated Datarun for each Hyperpartition or not.
                Optional. Defaults to ``False``.
            score_target (str):
                Which score to use for the tuning and selection process. It can be ``'cv'`` or
                ``'test'``. Optional. Defaults to ``'cv'``.
            priority (int):
                Priority of this Datarun. The higher the better. Optional. Defaults to ``1``.
            selector (str):
                Type of selector to use. Optional. Defaults to ``'uniform'``.
            tuner (str):
                Type of tuner to use. Optional. Defaults to ``'uniform'``.
            deadline (str):
                Time deadline. It must be a string representing a datetime in the format
                ``'%Y-%m-%d %H:%M'``. If given, ``budget_type`` will be set to ``'walltime'``.

        Returns:
            Datarun:
                The created Datarun or list of Dataruns.
        """

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

        dataruns = list()
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
        """Get unfinished Dataruns from the database and work on them.

        Check the ModelHub Database for unfinished Dataruns, and work on them
        as they are added. This process will continue to run until it exceeds
        total_time or there are no more Dataruns to process or it is killed.

        Args:
            datarun_ids (list):
                list of IDs of Dataruns to work on. If ``None``, this will work on any
                unfinished Dataruns found in the database. Optional. Defaults to ``None``.
            save_files (bool):
                Whether to save the fitted classifiers and their metrics or not.
                Optional. Defaults to True.
            choose_randomly (bool):
                If ``True``, work on all the highest-priority dataruns in random order.
                Otherwise, work on them in sequential order (by ID).
                Optional. Defaults to ``True``.
            cloud_mode (bool):
                Save the models and metrics in AWS S3 instead of locally. This option
                works only if S3 configuration has been provided on initialization.
                Optional. Defaults to ``False``.
            total_time (int):
                Total time to run the work process, in seconds. If ``None``, continue to
                run until interrupted or there are no more Dataruns to process.
                Optional. Defaults to ``None``.
            wait (bool):
                If ``True``, wait for more Dataruns to be inserted into the Database
                once all have been processed. Otherwise, exit the worker loop
                when they run out.
                Optional. Defaults to ``False``.
            verbose (bool):
                Whether to be verbose about the process. Optional. Defaults to ``True``.
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
                                 self._LOOP_WAIT)
                    time.sleep(self._LOOP_WAIT)
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
                LOGGER.error('Something went wrong. Sleeping %d seconds.', self._LOOP_WAIT)
                time.sleep(self._LOOP_WAIT)

            elapsed_time = (datetime.now() - start_time).total_seconds()
            if total_time is not None and elapsed_time >= total_time:
                LOGGER.info('Total run time for worker exceeded; exiting.')
                break

    def run(self, train_path, test_path=None, name=None, description=None,
            class_column='class', budget=100, budget_type='classifier', gridding=0, k_window=3,
            metric='f1', methods=['logreg', 'dt', 'knn'], r_minimum=2, run_per_partition=False,
            score_target='cv', selector='uniform', tuner='uniform', deadline=None, priority=1,
            save_files=True, choose_randomly=True, cloud_mode=False, total_time=None,
            verbose=True):

        """Create a Dataset and a Datarun and then work on it.

        Args:
            train_path (str):
                Path to the training CSV file. It can be a local filesystem path,
                absolute or relative, or an HTTP or HTTPS URL, or an S3 path in the
                format ``s3://{bucket_name}/{key}``. Required.
            test_path (str):
                Path to the testing CSV file. It can be a local filesystem path,
                absolute or relative, or an HTTP or HTTPS URL, or an S3 path in the
                format ``s3://{bucket_name}/{key}``.
                Optional. If not given, the training CSV will be split in two parts,
                train and test.
            name (str):
                Name given to this dataset. Optional. If not given, a hash will be
                generated from the training_path and used as the Dataset name.
            description (str):
                Human friendly description of the Dataset. Optional.
            class_column (str):
                Name of the column that will be used as the target variable.
                Optional. Defaults to ``'class'``.
            budget (int):
                Budget amount. Optional. Defaults to ``100``.
            budget_type (str):
                Budget Type. Can be 'classifier' or 'walltime'.
                Optional. Defaults to ``'classifier'``.
            gridding (int):
                ``gridding`` setting for the Tuner. Optional. Defaults to ``0``.
            k_window (int):
                ``k`` setting for the Selector. Optional. Defaults to ``3``.
            metric (str):
                Metric to use for the tuning and selection. Optional. Defaults to ``'f1'``.
            methods (list):
                List of methods to try. Optional. Defaults to ``['logreg', 'dt', 'knn']``.
            r_minimum (int):
                ``r_minimum`` setting for the Tuner. Optional. Defaults to ``2``.
            run_per_partition (bool):
                whether to create a separated Datarun for each Hyperpartition or not.
                Optional. Defaults to ``False``.
            score_target (str):
                Which score to use for the tuning and selection process. It can be ``'cv'`` or
                ``'test'``. Optional. Defaults to ``'cv'``.
            priority (int):
                Priority of this Datarun. The higher the better. Optional. Defaults to ``1``.
            selector (str):
                Type of selector to use. Optional. Defaults to ``'uniform'``.
            tuner (str):
                Type of tuner to use. Optional. Defaults to ``'uniform'``.
            deadline (str):
                Time deadline. It must be a string representing a datetime in the format
                ``'%Y-%m-%d %H:%M'``. If given, ``budget_type`` will be set to ``'walltime'``.
            verbose (bool):
                Whether to be verbose about the process. Optional. Defaults to ``True``.

        Returns:
            Datarun:
                The created Datarun or list of Dataruns.
        """

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
        """Load a Model from the Database.

        Args:
            classifier_id (int):
                Id of the Model to load.

        Returns:
            Model:
                The loaded model instance.
        """
        return self.db.get_classifier(classifier_id).load_model()
