CREATE TABLE `algorithms` (
  `id` integer NOT NULL PRIMARY KEY AUTOINCREMENT, 
  `code` varchar(15) DEFAULT ('') NOT NULL, 
  `name` varchar(30) DEFAULT ('') NOT NULL, 
  `probability` boolean NOT NULL
);

CREATE TABLE `dataruns` (
  `id` integer NOT NULL PRIMARY KEY AUTOINCREMENT,
  `name` varchar(100) DEFAULT ('') NOT NULL,
  `description` varchar(200) DEFAULT ('') NOT NULL,
  `dataset_description` varchar(1000) DEFAULT ('') NOT NULL,
  `trainpath` varchar(200) DEFAULT ('') NOT NULL,
  `testpath` varchar(200) DEFAULT ('') NOT NULL,
  `local_trainpath` varchar(300),
  `local_testpath` varchar(300),
  `datawrapper` text NOT NULL,
  `labelcol` bigint NOT NULL,
  `n` bigint NOT NULL,
  `k` bigint NOT NULL,
  `d` bigint NOT NULL,
  `majority` numeric(10, 9) NOT NULL,
  `size_kb` bigint NOT NULL,
  `sample_selection` varchar(255) DEFAULT ('uniform') NOT NULL,
  `frozen_selection` varchar(255) DEFAULT ('uniform') NOT NULL,
  `gridding` integer DEFAULT (0) NOT NULL,
  `priority` integer DEFAULT (5),
  `started` timestamp,
  `completed` timestamp,
  `budget` varchar(255) DEFAULT ('none'),
  `learner_budget` bigint,
  `walltime_budget_minutes` bigint,
  `deadline` timestamp,
  `metric` varchar(255),
  `score_target` varchar(255),
  `k_window` integer,
  `r_min` integer, 
  `is_gridding_done` boolean DEFAULT (0) NOT NULL,
  CHECK (`labelcol` >= 0),
  CHECK (`n` >= 0),
  CHECK (`k` >= 0),
  CHECK (`d` >= 0),
  CHECK (`size_kb` >= 0),
  CHECK (`learner_budget` >= 0),
  CHECK (`walltime_budget_minutes` >= 0)
);

CREATE TABLE `frozen_sets` (
  `id` integer NOT NULL PRIMARY KEY AUTOINCREMENT,
  `datarun_id` bigint,
  `algorithm` varchar(15) DEFAULT ('') NOT NULL,
  `trained` bigint DEFAULT (0),
  `rewards` numeric(20, 10) DEFAULT (0.0),
  `is_gridding_done` boolean DEFAULT (0) NOT NULL,
  `optimizables64` text,
  `constants64` text,
  `frozens64` text,
  `frozen_hash` varchar(32),
  CHECK (`datarun_id` >= 0),
  CHECK (`trained` >= 0)
);

CREATE TABLE `learners` (
  `id` integer NOT NULL PRIMARY KEY AUTOINCREMENT,
  `frozen_set_id` bigint,
  `datarun_id` integer,
  `dataname` varchar(100) DEFAULT ('') NOT NULL,
  `description` varchar(200) DEFAULT (''),
  `trainpath` varchar(300),
  `testpath` varchar(300),
  `modelpath` varchar(300),
  `metricpath` varchar(300),
  `params64` text NOT NULL,
  `trainable_params64` text,
  `dimensions` bigint,
  `cv_judgment_metric` numeric(20, 10),
  `cv_judgment_metric_stdev` numeric(20, 10),
  `test_judgment_metric` numeric(20, 10),
  `started` timestamp,
  `completed` timestamp,
  `status` varchar(255) DEFAULT 'started' NOT NULL,
  `error_msg` text,
  `host` varchar(50),
  CHECK (`frozen_set_id` >= 0),
  CHECK (`dimensions` >= 0)
);

CREATE INDEX `dataruns_name_desc_unq` ON `dataruns` (`name`, `description`);
