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

# Dump of table datasets
# ------------------------------------------------------------

DROP TABLE IF EXISTS `datasets`;

CREATE TABLE `datasets` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL DEFAULT '',
  `description` varchar(1000) NOT NULL DEFAULT '',
  `train_path` varchar(200) NOT NULL DEFAULT '',
  `test_path` varchar(200) NOT NULL DEFAULT '',
  `wrapper` longtext NOT NULL,
  `label_column` int(11) unsigned NOT NULL,
  `n_examples` int(11) unsigned NOT NULL,
  `k_classes` int(11) unsigned NOT NULL,
  `d_features` int(11) unsigned NOT NULL,
  `majority` decimal(10,9) NOT NULL,
  `size_kb` int(11) unsigned NOT NULL,
  PRIMARY KEY (`id`),
) ENGINE=MyISAM DEFAULT CHARSET=latin1;



# Dump of table dataruns
# ------------------------------------------------------------

DROP TABLE IF EXISTS `dataruns`;

CREATE TABLE `dataruns` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `dataset_id` int(11) unsigned DEFAULT NULL,
  `description` varchar(200) NOT NULL DEFAULT '',
  `tuner` enum('uniform', 'gp', 'gp_ei', 'gp_eivel', 'custom') NOT NULL DEFAULT 'uniform',
  `selector` enum('uniform', 'ucb1', 'bestk', 'bestkvel', 'recentk', 'recentkvel', 'hieralg', 'hierrand', 'purebestkvel') NOT NULL DEFAULT 'uniform',
  `gridding` int(11) unsigned NOT NULL default '0',
  `priority` smallint(10) DEFAULT '5',
  `started` datetime DEFAULT NULL,
  `completed` datetime DEFAULT NULL,
  `budget_type` enum('none', 'walltime', 'learner') DEFAULT 'none',
  `budget` int(11) unsigned DEFAULT NULL,
  `deadline` datetime DEFAULT NULL,
  `metric` enum('f1', 'f1_micro', 'f1_macro', 'f1_mu_sigma', 'roc_auc', 'roc_auc_micro', 'roc_auc_macro', 'accuracy') DEFAULT NULL,
  `score_target` enum('cv', 'test') DEFAULT NULL,
  `k_window` int(11) DEFAULT NULL,
  `r_min` int(11) DEFAULT NULL,
  `status` enum('pending', 'running', 'done') DEFAULT 'pending',
  PRIMARY KEY (`id`),
  KEY `name_desc_unq` (`name`, `description`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;



# Dump of table frozen_sets
# ------------------------------------------------------------

DROP TABLE IF EXISTS `frozen_sets`;

CREATE TABLE `frozen_sets` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `datarun_id` int(11) unsigned DEFAULT NULL,
  `algorithm` varchar(15) NOT NULL DEFAULT '',
  `trained` int(11) unsigned DEFAULT '0',
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

/* TODO: make the "judgment_metric" fields consistent with rest of code */

CREATE TABLE `learners` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `frozen_set_id` int(11) unsigned DEFAULT NULL,
  `datarun_id` int(11) unsigned DEFAULT NULL,
  `dataname` varchar(100) NOT NULL DEFAULT '',
  `description` varchar(200) NOT NULL DEFAULT '',
  `modelpath` varchar(300) DEFAULT NULL,
  `metricpath` varchar(300) DEFAULT NULL,
  `params64` mediumtext NOT NULL,
  `trainable_params64` longtext,
  `dimensions` int(11) unsigned DEFAULT NULL,
  `cv_judgment_metric` decimal(20,10) DEFAULT NULL,
  `cv_judgment_metric_stdev` decimal(20,10) DEFAULT NULL,
  `test_judgment_metric` decimal(20,10) DEFAULT NULL,
  `started` datetime DEFAULT NULL,
  `completed` datetime DEFAULT NULL,
  `status` enum('started', 'errored', 'complete') NOT NULL DEFAULT 'started',
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
