# ************************************************************
# Sequel Pro SQL dump
# Version 4096
#
# http://www.sequelpro.com/
# http://code.google.com/p/sequel-pro/
#
# Host: sql.mit.edu (MySQL 5.1.66-0+squeeze1-log)
# Database: drevo+delphi
# Generation Time: 2014-04-23 03:44:33 +0000
# ************************************************************


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;


# Dump of table algorithms
# ------------------------------------------------------------

DROP TABLE IF EXISTS `algorithms`;

CREATE TABLE `algorithms` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `code` varchar(15) NOT NULL DEFAULT '',
  `name` varchar(30) NOT NULL DEFAULT '',
  `probability` tinyint(1) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

LOCK TABLES `algorithms` WRITE;
/*!40000 ALTER TABLE `algorithms` DISABLE KEYS */;

INSERT INTO `algorithms` (`id`, `code`, `name`, `probability`)
VALUES
	(1,'classify_svm','Support Vector Machine', 1),
	(2,'classify_et','Extra Trees', 1),
	(3,'classify_pa','Passive Aggressive', 0),
	(4,'classify_sgd','Stochastic Gradient Descent', 1),
	(5,'classify_rf','Random Forest', 1),
	(6,'classify_mnb','Multinomial Naive Bayes', 1),
	(7,'classify_bnb','Bernoulii Naive Bayes', 1),
	(8,'classify_dbn','Deef Belief Network', 0),
	(9,'classify_logreg','Logistic Regression', 1),
	(10,'classify_gnb','Gaussian Naive Bayes', 1),
	(11,'classify_dt','Decision Tree', 1),
	(12,'classify_knn','K Nearest Neighbors', 1),
	(13,'classify_mlp','Multi-Layer Perceptron', 1),
	(14,'classify_gp','Gaussian Process', 1);

/*!40000 ALTER TABLE `algorithms` ENABLE KEYS */;
UNLOCK TABLES;


# Dump of table dataruns
# ------------------------------------------------------------

DROP TABLE IF EXISTS `dataruns`;

CREATE TABLE `dataruns` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL DEFAULT '',
  `description` varchar(200) NOT NULL DEFAULT '',
  `dataset_description` varchar(1000) NOT NULL DEFAULT '',
  `trainpath` varchar(200) NOT NULL DEFAULT '',
  `testpath` varchar(200) NOT NULL DEFAULT '',
  `local_trainpath` varchar(300) DEFAULT NULL,
  `local_testpath` varchar(300) DEFAULT NULL,
  `datawrapper` longtext NOT NULL,
  `labelcol` int(11) unsigned NOT NULL,
  `n` int(11) unsigned NOT NULL,
  `k` int(11) unsigned NOT NULL,
  `d` int(11) unsigned NOT NULL,
  `majority` decimal(10,9) NOT NULL,
  `size_kb` int(11) unsigned NOT NULL,
  `sample_selection` enum('uniform','gp', 'gp_ei', 'gp_eitime', 'gp_eivel', 'grid') NOT NULL DEFAULT 'uniform',
  `frozen_selection` enum('uniform','ucb1', 'bestk', 'bestkvel', 'recentk', 'recentkvel', 'hieralg', 'hierrand', 'purebestkvel') NOT NULL DEFAULT 'uniform',
  `priority` smallint(10) DEFAULT '5',
  `started` datetime DEFAULT NULL,
  `completed` datetime DEFAULT NULL,
  `budget` enum('none','walltime','learner') DEFAULT 'none',
  `learner_budget` int(11) unsigned DEFAULT NULL,
  `walltime_budget_minutes` int(11) unsigned DEFAULT NULL,
  `deadline` datetime DEFAULT NULL,
  `metric` enum('cv','test') DEFAULT NULL,
  `k_window` int(11) DEFAULT NULL,
  `r_min` int(11) DEFAULT NULL,
  `is_gridding_done` tinyint(1) NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `name_desc_unq` (`name`,`description`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;



# Dump of table frozen_sets
# ------------------------------------------------------------

DROP TABLE IF EXISTS `frozen_sets`;

CREATE TABLE `frozen_sets` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `datarun_id` int(11) unsigned DEFAULT NULL,
  `algorithm` varchar(15) NOT NULL DEFAULT '',
  `trained` int(11) unsigned DEFAULT '0',
  `rewards` decimal(20,10) unsigned DEFAULT '0.0000000000',
  `is_gridding_done` tinyint(1) NOT NULL DEFAULT '0',
  `optimizables64` longtext,
  `constants64` longtext,
  `frozens64` longtext,
  `frozen_hash` varchar(32) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;



# Dump of table learners
# ------------------------------------------------------------

DROP TABLE IF EXISTS `learners`;

CREATE TABLE `learners` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `frozen_set_id` int(11) unsigned DEFAULT NULL,
  `datarun_id` int(11) DEFAULT NULL,
  `dataname` varchar(100) NOT NULL DEFAULT '',
  `description` varchar(200) NULL DEFAULT '',
  `algorithm` varchar(15) NOT NULL DEFAULT '',
  `trainpath` varchar(300) DEFAULT NULL,
  `testpath` varchar(300) DEFAULT NULL,
  `modelpath` varchar(300) DEFAULT NULL,
  `params_hash` varchar(32) NOT NULL DEFAULT '',
  `frozen_hash` varchar(32) NOT NULL DEFAULT '',
  `datarun_name_hash` varchar(32) DEFAULT NULL,
  `params64` mediumtext NOT NULL,
  `trainable_params64` longtext,
  `dimensions` int(11) unsigned DEFAULT NULL,
  `cv` decimal(20,10) DEFAULT NULL,
  `stdev` decimal(20,10) DEFAULT NULL,
  `test` decimal(20,10) DEFAULT NULL,
  `cv_f1_scores64` longtext,
  `cv_pr_curve_aucs64` longtext,
  `cv_roc_curve_aucs64` longtext,
  `cv_pr_curve_precisions64` longtext,
  `cv_pr_curve_recalls64` longtext,
  `cv_pr_curve_thresholds64` longtext,
  `cv_roc_curve_fprs64` longtext,
  `cv_roc_curve_tprs64` longtext,
  `cv_roc_curve_thresholds64` longtext,
  `cv_rank_accuracies64` longtext,
  `test_f1_scores64` longtext,
  `test_pr_curve_aucs64` longtext,
  `test_roc_curve_aucs64` longtext,
  `test_pr_curve_precisions64` longtext,
  `test_pr_curve_recalls64` longtext,
  `test_pr_curve_thresholds64` longtext,
  `test_roc_curve_fprs64` longtext,
  `test_roc_curve_tprs64` longtext,
  `test_roc_curve_thresholds64` longtext,
  `test_rank_accuracies64` longtext,
  `confusion64` mediumtext,
  `started` datetime DEFAULT NULL,
  `completed` datetime DEFAULT NULL,
  `seconds` int(11) unsigned DEFAULT NULL,
  `errored` datetime DEFAULT NULL,
  `is_error` tinyint(1) NOT NULL DEFAULT '0',
  `error_msg` longtext,
  `host` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;




/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
