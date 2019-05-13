#!/usr/bin/python2.7
from __future__ import absolute_import, unicode_literals

import datetime
import imp
import logging
import os
import re
import traceback
import warnings
from builtins import object, str
from collections import defaultdict

import boto3
import numpy as np

from atm.classifier import Model
from atm.config import LogConfig
from atm.constants import CUSTOM_CLASS_REGEX, SELECTORS_MAP, TUNERS_MAP
from atm.database import ClassifierStatus, db_session
from atm.utilities import ensure_directory, get_instance, save_metrics, save_model, update_params

# shhh
warnings.filterwarnings('ignore')

# for garrays
os.environ['GNUMPY_IMPLICIT_CONVERSION'] = 'allow'

# load the library-wide logger
LOGGER = logging.getLogger('atm')


# Exception thrown when something goes wrong for the worker, but the worker
# handles the error.
class ClassifierError(Exception):
    pass


class Worker(object):
    def __init__(self, database, datarun, save_files=True, cloud_mode=False,
                 aws_config=None, log_config=None, public_ip='localhost'):
        """
        database: Database object with connection information
        datarun: Datarun ORM object to work on.
        save_files: if True, save model and metrics files to disk or cloud.
        cloud_mode: if True, save classifiers to the cloud
        aws_config: S3Config object with amazon s3 connection info
        """
        self.db = database
        self.datarun = datarun
        self.save_files = save_files
        self.cloud_mode = cloud_mode
        self.aws_config = aws_config
        self.public_ip = public_ip

        log_config = log_config or LogConfig({})
        self.model_dir = log_config.model_dir
        self.metric_dir = log_config.metric_dir
        self.verbose_metrics = log_config.verbose_metrics
        ensure_directory(self.model_dir)
        ensure_directory(self.metric_dir)

        # load the Dataset from the database
        self.dataset = self.db.get_dataset(self.datarun.dataset_id)

        # load the Selector and Tuner classes specified by our datarun
        self.load_selector()
        self.load_tuner()

    def load_selector(self):
        """
        Load and initialize the BTB class which will be responsible for
        selecting hyperpartitions.
        """
        # selector will either be a key into SELECTORS_MAP or a path to
        # a file that defines a class called CustomSelector.
        if self.datarun.selector in SELECTORS_MAP:
            Selector = SELECTORS_MAP[self.datarun.selector]
        else:
            path, classname = re.match(CUSTOM_CLASS_REGEX,
                                       self.datarun.selector).groups()
            mod = imp.load_source('btb.selection.custom', path)
            Selector = getattr(mod, classname)

        LOGGER.info('Selector: %s' % Selector)

        # generate the arguments we need to initialize the selector
        hyperpartitions = self.db.get_hyperpartitions(datarun_id=self.datarun.id)
        hp_by_method = defaultdict(list)
        for hp in hyperpartitions:
            hp_by_method[hp.method].append(hp.id)

        hyperpartition_ids = [hp.id for hp in hyperpartitions]

        # Selector classes support passing in redundant arguments
        self.selector = get_instance(Selector,
                                     choices=hyperpartition_ids,
                                     k=self.datarun.k_window,
                                     by_algorithm=dict(hp_by_method))

    def load_tuner(self):
        """
        Load, but don't initialize, the BTB class which will be responsible for
        choosing non-hyperpartition hyperparameter values (a subclass of Tuner). The
        tuner must be initialized with information about the hyperpartition, so it
        cannot be created until later.
        """
        # tuner will either be a key into TUNERS_MAP or a path to
        # a file that defines a class called CustomTuner.
        if self.datarun.tuner in TUNERS_MAP:
            self.Tuner = TUNERS_MAP[self.datarun.tuner]
        else:
            path, classname = re.match(CUSTOM_CLASS_REGEX, self.datarun.tuner).groups()
            mod = imp.load_source('btb.tuning.custom', path)
            self.Tuner = getattr(mod, classname)

        LOGGER.info('Tuner: %s' % self.Tuner)

    def select_hyperpartition(self):
        """
        Use the hyperpartition selection method specified by our datarun to choose a
        hyperpartition of hyperparameters from the ModelHub. Only consider
        partitions for which gridding is not complete.
        """
        hyperpartitions = self.db.get_hyperpartitions(datarun_id=self.datarun.id)

        # load classifiers and build scores lists
        # make sure all hyperpartitions are present in the dict, even ones that
        # don't have any classifiers. That way the selector can choose hyperpartitions
        # that haven't been scored yet.
        hyperpartition_scores = {fs.id: [] for fs in hyperpartitions}
        classifiers = self.db.get_classifiers(datarun_id=self.datarun.id)

        for c in classifiers:
            # ignore hyperpartitions for which gridding is done
            if c.hyperpartition_id not in hyperpartition_scores:
                continue

            # the cast to float is necessary because the score is a Decimal;
            # doing Decimal-float arithmetic throws errors later on.
            score = float(getattr(c, self.datarun.score_target) or 0)
            hyperpartition_scores[c.hyperpartition_id].append(score)

        hyperpartition_id = self.selector.select(hyperpartition_scores)
        return self.db.get_hyperpartition(hyperpartition_id)

    def tune_hyperparameters(self, hyperpartition):
        """
        Use the hyperparameter tuning method specified by our datarun to choose
        a set of hyperparameters from the potential space.
        """
        # Get parameter metadata for this hyperpartition
        tunables = hyperpartition.tunables

        # If there aren't any tunable parameters, we're done. Return the vector
        # of values in the hyperpartition and mark the set as finished.
        if not len(tunables):
            LOGGER.warning('No tunables for hyperpartition %d' % hyperpartition.id)
            self.db.mark_hyperpartition_gridding_done(hyperpartition.id)
            return update_params(params=[],
                                 tunables=tunables,
                                 categoricals=hyperpartition.categoricals,
                                 constants=hyperpartition.constants)

        # Get previously-used parameters: every classifier should either be
        # completed or have thrown an error
        all_clfs = self.db.get_classifiers(hyperpartition_id=hyperpartition.id)
        classifiers = [c for c in all_clfs if c.status == ClassifierStatus.COMPLETE]

        X = [c.hyperparameter_values for c in classifiers]
        y = np.array([float(getattr(c, self.datarun.score_target)) for c in classifiers])

        # Initialize the tuner and propose a new set of parameters
        # this has to be initialized with information from the hyperpartition, so we
        # need to do it fresh for each classifier (not in load_tuner)
        tuner = get_instance(self.Tuner,
                             tunables=tunables,
                             gridding=self.datarun.gridding,
                             r_minimum=self.datarun.r_minimum)
        if len(X) > 0:
            tuner.add(X, y)

        params = tuner.propose()
        if params is None and self.datarun.gridding:
            LOGGER.info('Gridding done for hyperpartition %d' % hyperpartition.id)
            self.db.mark_hyperpartition_gridding_done(hyperpartition.id)
            return None

        # Append categorical and constants to the params.
        return update_params(params=params,
                             categoricals=hyperpartition.categoricals,
                             constants=hyperpartition.constants)

    def test_classifier(self, method, params):
        """
        Given a set of fully-qualified hyperparameters, create and test a
        classifier model.
        Returns: Model object and metrics dictionary
        """
        model = Model(method=method, params=params,
                      judgment_metric=self.datarun.metric,
                      class_column=self.dataset.class_column,
                      verbose_metrics=self.verbose_metrics)

        train_path, test_path = self.dataset.load_()

        metrics = model.train_test(self.dataset)

        target = self.datarun.score_target

        def metric_string(model):
            if 'cv' in target or 'mu_sigma' in target:
                return '%.3f +- %.3f' % (model.cv_judgment_metric,
                                         2 * model.cv_judgment_metric_stdev)
            else:
                return '%.3f' % model.test_judgment_metric

        LOGGER.info('Judgment metric (%s, %s): %s' % (self.datarun.metric,
                                                      target[:-len('_judgment_metric')],
                                                      metric_string(model)))

        old_best = self.db.get_best_classifier(datarun_id=self.datarun.id,
                                               score_target=target)
        if old_best is not None:
            if getattr(model, target) > getattr(old_best, target):
                LOGGER.info('New best score! Previous best (classifier %s): %s',
                            old_best.id, metric_string(old_best))
            else:
                LOGGER.info('Best so far (classifier %s): %s',
                            old_best.id, metric_string(old_best))

        return model, metrics

    def save_classifier(self, classifier_id, model, metrics):
        """
        Update a classifier with metrics and model information and mark it as
        "complete"

        classifier_id: ID of the classifier to save
        model: Model object containing a serializable representation of the
            final model generated by this classifier.
        metrics: Dictionary containing cross-validation and test metrics data
            for the model.
        """
        # whether to save model and metrics data to the filesystem
        if self.save_files:
            # keep a database session open so that the utility functions can
            # access the linked hyperpartitions and dataruns
            with db_session(self.db):
                classifier = self.db.get_classifier(classifier_id)
                model_path = save_model(classifier, self.model_dir, model)
                metric_path = save_metrics(classifier, self.metric_dir, metrics)

            # if necessary, save model and metrics to Amazon S3 bucket
            if self.cloud_mode:
                try:
                    self.save_classifier_cloud(model_path, metric_path)

                except Exception:
                    msg = traceback.format_exc()
                    LOGGER.error('Error in save_classifier_cloud()')
                    self.db.mark_classifier_errored(classifier_id, error_message=msg)
        else:
            model_path = None
            metric_path = None

        # update the classifier in the database
        self.db.complete_classifier(classifier_id=classifier_id,
                                    model_location=model_path,
                                    metrics_location=metric_path,
                                    cv_score=model.cv_judgment_metric,
                                    cv_stdev=model.cv_judgment_metric_stdev,
                                    test_score=model.test_judgment_metric)

        # update this session's hyperpartition entry
        LOGGER.info('Saved classifier %d.' % classifier_id)

    def save_classifier_cloud(self, local_model_path, local_metric_path, delete_local=False):
        """
        Save a classifier to the S3 bucket supplied by aws_config. Saves a
        serialized representaion of the model as well as a detailed set
        of metrics.

        local_model_path: path to serialized model in the local file system
        local_metric_path: path to serialized metrics in the local file system
        """

        aws_access_key = None
        aws_secret_key = None

        if self.aws_config:
            aws_access_key = self.aws_config.access_key
            aws_secret_key = self.aws_config.secret_key

        client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )

        if self.aws_config.aws_folder:
            aws_model_path = os.path.join(self.aws_config.aws_folder, local_model_path)
            aws_metric_path = os.path.join(self.aws_config.aws_folder, local_metric_path)

        else:
            aws_model_path = local_model_path
            aws_metric_path = local_metric_path

        client.upload_file(aws_model_path, self.aws_config.bucket, aws_model_path)
        LOGGER.info('Uploading model at %s to S3 bucket %s',
                    self.aws_config.s3_bucket, local_model_path)

        client.upload_file(aws_metric_path, self.aws_config.bucket, aws_metric_path)
        LOGGER.info('Uploading metrics at %s to S3 bucket %s',
                    self.aws_config.s3_bucket, local_metric_path)

        if delete_local:
            LOGGER.info('Deleting local copies of %s and %s',
                        local_model_path, local_metric_path)
            os.remove(local_model_path)
            os.remove(local_metric_path)

    def is_datarun_finished(self):
        """
        Check to see whether the datarun is finished. This could be due to the
        budget being exhausted or due to hyperparameter gridding being done.
        """
        hyperpartitions = self.db.get_hyperpartitions(datarun_id=self.datarun.id)
        if not hyperpartitions:
            LOGGER.warning('No incomplete hyperpartitions for datarun %d present in database.'
                           % self.datarun.id)
            return True

        if self.datarun.budget_type == 'classifier':
            # hyperpartition classifier counts are updated whenever a classifier
            # is created, so this will count running, errored, and complete.
            n_completed = len(self.db.get_classifiers(datarun_id=self.datarun.id))
            if n_completed >= self.datarun.budget:
                LOGGER.warning('Classifier budget has run out!')
                return True

        elif self.datarun.budget_type == 'walltime':
            deadline = self.datarun.deadline
            if datetime.datetime.now() > deadline:
                LOGGER.warning('Walltime budget has run out!')
                return True

        return False

    def run_classifier(self, hyperpartition_id=None):
        """
        Choose hyperparameters, then use them to test and save a Classifier.
        """
        # check to see if our work is done
        if self.is_datarun_finished():
            # marked the run as done successfully
            self.db.mark_datarun_complete(self.datarun.id)
            LOGGER.warning('Datarun %d has ended.' % self.datarun.id)
            return

        try:
            LOGGER.debug('Choosing hyperparameters...')
            if hyperpartition_id is not None:
                hyperpartition = self.db.get_hyperpartition(hyperpartition_id)
                if hyperpartition.datarun_id != self.datarun.id:
                    LOGGER.error('Hyperpartition %d is not a part of datarun %d'
                                 % (hyperpartition_id, self.datarun.id))
                    return
            else:
                # use the multi-arm bandit to choose which hyperpartition to use next
                hyperpartition = self.select_hyperpartition()

            # use tuner to choose a set of parameters for the hyperpartition
            params = self.tune_hyperparameters(hyperpartition)

        except Exception:
            LOGGER.error('Error choosing hyperparameters: datarun=%s' % str(self.datarun))
            LOGGER.error(traceback.format_exc())
            raise ClassifierError()

        if params is None:
            LOGGER.warning('No parameters chosen: hyperpartition %d is finished.'
                           % hyperpartition.id)
            return

        param_info = 'Chose parameters for method "%s":' % hyperpartition.method
        for k in sorted(params.keys()):
            param_info += '\n\t%s = %s' % (k, params[k])

        LOGGER.info(param_info)

        LOGGER.debug('Creating classifier...')
        classifier = self.db.start_classifier(hyperpartition_id=hyperpartition.id,
                                              datarun_id=self.datarun.id,
                                              host=self.public_ip,
                                              hyperparameter_values=params)

        try:
            LOGGER.debug('Testing classifier...')
            model, metrics = self.test_classifier(hyperpartition.method, params)
            LOGGER.debug('Saving classifier...')
            self.save_classifier(classifier.id, model, metrics)

        except Exception:
            msg = traceback.format_exc()
            LOGGER.error('Error testing classifier: datarun=%s' % str(self.datarun))
            LOGGER.error(msg)
            self.db.mark_classifier_errored(classifier.id, error_message=msg)
            raise ClassifierError()
